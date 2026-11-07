"""
The agent loop: the LLM decides, the tools do the work.

This is the heart of the project, and the thing worth being able to explain.

A normal chatbot gets a question and produces text in one shot. It has no way
to look anything up, so when it does not know a number it invents one.

An agent is given a question *and a list of tools*. Then this loop runs:

    1. send the conversation so far to the model, along with the tool list
    2. the model replies either with a tool call, or with a final answer
    3. if it is a tool call: run the tool, append the result to the
       conversation, and go back to step 1
    4. if it is a final answer: stop

Two consequences matter:

* **Self-correction.** When run_sql fails, the error text goes back into the
  conversation as the tool's reply. The model reads "Referenced column
  'revenue' not found" and writes a corrected query on its next turn. Nobody
  programmed that recovery; it falls out of the loop.

* **No invented numbers.** The system prompt forbids any figure that did not
  come from a query result. Combined with the loop, the model's only route to
  a number is to actually run SQL -- so the answer is grounded in the data.

The loop is capped at MAX_AGENT_STEPS turns so a confused model cannot spin
forever; when the cap is hit the user gets an honest "I could not work this
out" rather than a guess.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from src import config, logger
from src.data_loader import DataStore
from src.executor import QueryResult
from src.schema import build_schema_text, schema_summary
from src.tools import ChartRequest, ToolBox, ToolCallRecord

# The exact wording the model is told to use when the data cannot answer the
# question. Keeping it fixed means the UI and the evaluation can both detect a
# refusal reliably, instead of guessing from free text.
REFUSAL_PHRASE = "cannot be answered from this data"


class AgentError(Exception):
    """Something went wrong talking to the model (bad key, no network, ...)."""


SYSTEM_PROMPT = """You are DataPilot, a careful data analyst.

You answer questions about the SQL tables described below by writing and
running SQL. You have no other source of information.

=== THE DATA YOU CAN QUERY ===
{schema}
=== END OF DATA DESCRIPTION ===

RULES, in order of importance:

1. Use only the tables and columns listed above, spelled exactly as shown.
   Never invent a table, a column, or a category value. If you are unsure
   what a column contains, call describe_table before writing SQL.

2. Every number, name and date in your answer must come from a run_sql result
   in this conversation. Never estimate, never work it out in your head, and
   never rely on what you remember. No query run means no answer available.

3. The database is DuckDB and strictly read-only. Only SELECT queries work
   (they may begin with WITH). Anything that changes data will be refused.

4. If run_sql returns an error, read the error, correct the query, and call
   run_sql again. Common causes are a misspelled column, a category value
   that does not exist, and the wrong table.

5. If a measure the question needs is not stored directly, calculate it from
   the columns that do exist. For example, if there are quantity, unit_price
   and discount columns but no revenue column, revenue is
   quantity * unit_price * (1 - discount).

6. Interpret "last quarter", "last month", "recently" and similar relative
   wording against the newest date present in the data -- NOT against today's
   real-world date. Find that date with a query such as
   SELECT MAX(date_column) FROM the_table if you need it.

7. If the data genuinely cannot answer the question -- the column simply is not
   there, or the fact was never recorded -- reply with the exact phrase
   "{refusal_phrase}" and say in one short sentence what is missing. Do not
   guess, and do not quietly answer a different, easier question instead.

8. Call make_chart only when a chart genuinely adds something: comparing
   categories, or showing a change over time. Never chart a single number.
   Call it at most once, after you have the result you want to show.

9. When the answer is about a particular thing -- a region, a product, a
   customer, a month -- SELECT that identifying column alongside the number.
   A result table containing only a bare number does not show the reader what
   the number refers to.

10. Never invent a unit or a currency symbol. If the data does not record what
    currency an amount is in, give the plain number with no symbol in front
    of it.

11. Your final answer must be 1-3 plain sentences of ordinary prose that
    directly answer the question, including the actual figures. Write it as
    sentences, not as a list: no bullet points, no numbered lists, no bold
    markdown, no headings, no SQL and no tables. The app already shows the
    user the query and the result table underneath your answer.
"""


@dataclass
class AgentAnswer:
    """Everything one question produced, for the UI, the logs and the evaluation."""
    question: str
    answer: str
    success: bool
    sql: str | None = None
    result: QueryResult | None = None          # the query the answer rests on
    chart_request: ChartRequest | None = None
    chart_result: QueryResult | None = None    # the query the chart is drawn from
    steps: list[ToolCallRecord] = field(default_factory=list)
    llm_turns: int = 0
    self_corrections: int = 0
    duration_seconds: float = 0.0
    model: str = ""
    error: str | None = None
    refused: bool = False          # the agent said the data cannot answer this

    @property
    def dataframe(self):
        """The result table, or None."""
        return self.result.dataframe if self.result is not None else None

    @property
    def chart_is_from_another_query(self) -> bool:
        """
        True when the chart comes from an earlier query than the answer does.

        The UI says so under the chart, so the chart is never silently
        attributed to the SQL shown above it.
        """
        return (self.chart_result is not None
                and self.result is not None
                and self.chart_result is not self.result)

    def step_summary(self) -> list[str]:
        """One readable line per tool call, for the 'How I got this' panel."""
        return [
            f"{'ok  ' if call.ok else 'fail'}  {call.short_description()}"
            for call in self.steps
        ]


def _looks_like_a_refusal(answer: str) -> bool:
    """
    Decide whether the agent said the data cannot answer the question.

    The exact phrase from the prompt is checked first. The extra patterns are
    a safety net for when the model paraphrases, which it occasionally does.
    """
    lowered = answer.lower()
    if REFUSAL_PHRASE in lowered:
        return True
    patterns = [
        "cannot be answered", "can't be answered", "cannot answer",
        "not possible to answer", "no column", "not available in the data",
        "is not recorded", "does not contain", "doesn't contain",
        "there is no information", "no data about",
    ]
    return any(pattern in lowered for pattern in patterns)


def _tool_calls_as_dicts(message: Any) -> list[dict[str, Any]]:
    """
    Convert the model's tool calls into plain dictionaries.

    The conversation we send back must repeat the assistant's own tool calls
    exactly, or the API rejects the follow-up. Building plain dicts avoids
    depending on the SDK's object types.
    """
    calls = []
    for call in (message.tool_calls or []):
        calls.append({
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments or "{}",
            },
        })
    return calls


def _parse_arguments(raw: str) -> dict[str, Any]:
    """
    Read the JSON arguments the model produced.

    Models sometimes emit slightly malformed JSON. Rather than crash, we hand
    back an empty dict; the tool then reports a readable error, and the model
    gets a chance to try again -- the same self-correction path as a bad query.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# How long to wait before retrying after a rate limit, in seconds. Groq's free
# tier limits tokens per minute, so the waits have to be long enough for the
# window to roll over -- retrying after 200ms just burns the budget again.
RATE_LIMIT_BACKOFF = (3, 8, 20, 40)

# Give up on a single HTTP request after this long, rather than waiting on the
# SDK's much longer default.
REQUEST_TIMEOUT_SECONDS = 90.0


def _is_rate_limit(error: Exception) -> bool:
    """Spot a rate-limit rejection, whatever shape the SDK gives it."""
    if type(error).__name__ in {"RateLimitError", "TooManyRequests"}:
        return True
    if getattr(error, "status_code", None) == 429:
        return True
    text = str(error).lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


def _is_timeout(error: Exception) -> bool:
    """Spot a request that timed out or lost its connection."""
    if type(error).__name__ in {"APITimeoutError", "APIConnectionError",
                                "ReadTimeout", "ConnectTimeout", "Timeout"}:
        return True
    text = str(error).lower()
    return "timed out" in text or "timeout" in text or "connection error" in text


def _retry_after_seconds(error: Exception) -> float | None:
    """Use the server's own Retry-After hint when it sends one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            # Values arrive either as plain seconds or as "7.66s".
            return float(str(raw).rstrip("s"))
        except ValueError:
            continue
    return None


class Agent:
    """
    Runs one agent loop per question against one loaded dataset.

    Create it once per dataset and call ask() as often as you like; each call
    starts a fresh conversation, so questions do not contaminate each other.
    """

    def __init__(
        self,
        store: DataStore,
        api_key: str | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        include_value_hints: bool = True,
        max_rows: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.store = store
        self.model = model or config.model_name()
        self.max_steps = max_steps or config.max_agent_steps()
        self.include_value_hints = include_value_hints
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

        key = api_key or config.groq_api_key()
        if not key:
            raise AgentError(
                "No Groq API key found. Add GROQ_API_KEY to your .env file "
                "(or to Streamlit secrets when deployed). "
                "You can get a free key at https://console.groq.com/keys"
            )

        try:
            from groq import Groq
        except ImportError as error:  # pragma: no cover
            raise AgentError("The 'groq' package is not installed. Run: pip install groq") from error

        # An explicit per-request timeout matters: without one a stalled
        # request can hang for ten minutes and look like the app has frozen.
        # Found when an evaluation run sat idle for 17 minutes on a single
        # hung call. max_retries covers transient connection errors; rate
        # limits are handled separately in _call_model, because those need
        # much longer waits than the SDK's default.
        self._client = Groq(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=2)
        self.schema_text = build_schema_text(store, include_value_hints=include_value_hints)

    # -- the prompt ----------------------------------------------------------

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            schema=self.schema_text, refusal_phrase=REFUSAL_PHRASE
        )

    # -- talking to the model ------------------------------------------------

    def _call_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        """
        Send one request, retrying politely if we are rate limited.

        Groq's free tier caps tokens per minute. Without this, a burst of
        questions (the evaluation suite, or an enthusiastic user) turns into a
        wall of failures that look like bugs but are only impatience.
        """
        last_error: Exception | None = None

        for attempt in range(len(RATE_LIMIT_BACKOFF) + 1):
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,            # type: ignore[arg-type]
                    tools=tools,                  # type: ignore[arg-type]
                    tool_choice="auto",
                    temperature=0,                # same question -> same answer
                    max_tokens=2000,
                )
            except Exception as error:            # noqa: BLE001
                last_error = error
                retryable = _is_rate_limit(error) or _is_timeout(error)
                if not retryable or attempt == len(RATE_LIMIT_BACKOFF):
                    raise
                wait = _retry_after_seconds(error) or RATE_LIMIT_BACKOFF[attempt]
                time.sleep(min(wait + 0.5, 60))

        raise last_error                          # pragma: no cover

    # -- the loop ------------------------------------------------------------

    def ask(self, question: str, log: bool = True) -> AgentAnswer:
        """Answer one question. Never raises for ordinary failures."""
        started = time.perf_counter()
        question = str(question or "").strip()

        if not question:
            return AgentAnswer(
                question=question, answer="Please type a question first.",
                success=False, model=self.model,
            )

        toolbox = ToolBox.for_store(
            self.store,
            **{k: v for k, v in
               {"max_rows": self.max_rows, "timeout_seconds": self.timeout_seconds}.items()
               if v is not None},
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": question},
        ]

        final_answer = ""
        error_message: str | None = None
        turns = 0

        try:
            for turn in range(self.max_steps):
                turns = turn + 1

                try:
                    response = self._call_model(messages, toolbox.definitions())
                except Exception as error:            # noqa: BLE001
                    error_message = _readable_api_error(error)
                    break

                message = response.choices[0].message
                tool_calls = _tool_calls_as_dicts(message)

                # No tool calls means the model is done and this is the answer.
                if not tool_calls:
                    final_answer = (message.content or "").strip()
                    break

                # Repeat the assistant's turn back, then answer each tool call.
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": tool_calls,
                })

                for call in tool_calls:
                    name = call["function"]["name"]
                    arguments = _parse_arguments(call["function"]["arguments"])
                    record = toolbox.call(name, arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": record.observation,
                    })
            else:
                # The for-loop finished without ever breaking: out of steps.
                error_message = (
                    f"I could not work this out within {self.max_steps} steps. "
                    "Try asking in a simpler way, or splitting it into two questions."
                )

        finally:
            toolbox.close()

        duration = time.perf_counter() - started

        # Work out what to report.
        if error_message and not final_answer:
            answer_text = error_message
            success = False
        elif not final_answer:
            answer_text = "I could not produce an answer for that question."
            success = False
        else:
            answer_text = final_answer
            success = True

        refused = success and _looks_like_a_refusal(answer_text)

        # A successful answer that is not a refusal should rest on a query.
        # If the model answered without ever running SQL, that is a failure of
        # rule 2, and it is honest to say so rather than present a guess.
        if success and not refused and toolbox.last_result is None:
            answer_text = (
                "I could not answer that from the data: I did not manage to run a "
                "query that produced a result. Please try rephrasing the question."
            )
            success = False

        outcome = AgentAnswer(
            question=question,
            answer=answer_text,
            success=success,
            sql=toolbox.last_result.sql if toolbox.last_result else None,
            result=toolbox.last_result,
            chart_request=toolbox.chart_request,
            chart_result=toolbox.chart_result,
            steps=toolbox.calls,
            llm_turns=turns,
            self_corrections=toolbox.failed_sql_attempts,
            duration_seconds=duration,
            model=self.model,
            error=error_message,
            refused=refused,
        )

        if log:
            logger.log_question(
                question=outcome.question,
                answer=outcome.answer,
                sql=outcome.sql,
                success=outcome.success,
                duration_seconds=outcome.duration_seconds,
                steps=len(outcome.steps),
                self_corrections=outcome.self_corrections,
                tables=schema_summary(self.store),
                model=outcome.model,
                error=outcome.error,
                extra={"refused": outcome.refused, "llm_turns": outcome.llm_turns},
            )

        return outcome


def _readable_api_error(error: Exception) -> str:
    """Turn an SDK exception into something a user can act on."""
    text = str(error)
    lowered = text.lower()

    if "authentication" in lowered or "401" in lowered or "invalid api key" in lowered:
        return ("The Groq API key was rejected. Check GROQ_API_KEY in your .env file "
                "or in Streamlit secrets.")
    if "rate limit" in lowered or "429" in lowered:
        return "Groq's rate limit was hit. Wait a few seconds and ask again."
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered
                               or "decommissioned" in lowered):
        return (f"The model in MODEL_NAME is not available. Pick a current one from "
                f"https://console.groq.com/docs/models  (details: {text[:120]})")
    if "connection" in lowered or "timeout" in lowered or "network" in lowered:
        return "Could not reach Groq. Check your internet connection and try again."

    return f"The model could not be reached: {text[:200]}"
