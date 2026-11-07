"""
Tests for the agent loop (Phase 3).

    python tests/test_agent.py              # fast, no API key needed
    python tests/test_agent.py --live       # also asks the real model 5 questions

The interesting tests here use a *fake* LLM instead of the real one. That is
deliberate. Self-correction is the headline feature, and "run it and hope the
model makes a mistake" is not a test. With a scripted fake model I can force
the exact sequence I want -- a broken query, then a fixed one -- and prove the
loop recovers, every single time, in under a second and for free.

The --live run then checks the real thing end to end.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import logger                                        # noqa: E402
from src.agent import REFUSAL_PHRASE, Agent, _looks_like_a_refusal  # noqa: E402
from src.data_loader import load_sample_dataset               # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  {detail}")


# ---------------------------------------------------------------------------
# A fake model, so we can script exactly what it replies
# ---------------------------------------------------------------------------

@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] = field(default_factory=list)


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


class FakeClient:
    """
    Stands in for the Groq client.

    Give it a list of replies; it hands them out one per turn. Each reply is
    either ("tool", name, arguments_dict) or ("text", the final answer).
    """

    def __init__(self, replies: list[tuple]) -> None:
        self.replies = list(replies)
        self.turns = 0
        self.conversations: list[list[dict[str, Any]]] = []

        fake_client = self

        class Completions:
            def create(self, **kwargs: Any) -> FakeResponse:
                fake_client.turns += 1
                fake_client.conversations.append(list(kwargs.get("messages", [])))

                if not fake_client.replies:
                    return FakeResponse([FakeChoice(FakeMessage(content="(no more replies)"))])

                reply = fake_client.replies.pop(0)
                if reply[0] == "tool":
                    import json
                    _, name, arguments = reply
                    call = FakeToolCall(
                        id=f"call_{fake_client.turns}",
                        function=FakeFunction(name=name, arguments=json.dumps(arguments)),
                    )
                    return FakeResponse([FakeChoice(FakeMessage(content="", tool_calls=[call]))])
                return FakeResponse([FakeChoice(FakeMessage(content=reply[1]))])

        class Chat:
            completions = Completions()

        self.chat = Chat()


def make_agent(store, replies: list[tuple], **options: Any) -> tuple[Agent, FakeClient]:
    """Build an Agent wired to a scripted fake model."""
    agent = Agent(store, api_key="test-key-not-used", **options)
    fake = FakeClient(replies)
    agent._client = fake            # swap the real client for the fake one
    return agent, fake


# ---------------------------------------------------------------------------
# 1. Self-correction
# ---------------------------------------------------------------------------

def test_self_correction(store) -> None:
    print("\n1. The agent fixes its own broken SQL")

    agent, fake = make_agent(store, [
        # Turn 1: a query using a column that does not exist.
        ("tool", "run_sql", {"query": "SELECT SUM(revenue) AS total FROM sales"}),
        # Turn 2: having seen the error, a correct query.
        ("tool", "run_sql", {"query":
            "SELECT SUM(quantity * unit_price * (1 - discount)) AS total FROM sales"}),
        # Turn 3: the final answer.
        ("text", "Total revenue is 103,437,000 rupees."),
    ])
    outcome = agent.ask("What is total revenue?", log=False)

    check("it succeeded in the end", outcome.success, outcome.answer)
    check("one self-correction was counted", outcome.self_corrections == 1,
          f"got {outcome.self_corrections}")
    check("it took two SQL attempts", len(outcome.steps) == 2, f"got {len(outcome.steps)}")
    check("the first attempt is recorded as failed", not outcome.steps[0].ok)
    check("the second attempt is recorded as succeeded", outcome.steps[1].ok)
    check("the SQL reported is the working one",
          outcome.sql is not None and "unit_price" in outcome.sql, str(outcome.sql))
    check("a result table came back", outcome.result is not None and outcome.result.row_count == 1)

    # The important bit: the error text was actually fed back to the model.
    last_conversation = fake.conversations[-1]
    tool_replies = [m for m in last_conversation if m.get("role") == "tool"]
    first_reply = tool_replies[0]["content"] if tool_replies else ""
    check("the error message was fed back to the model",
          "ERROR" in first_reply and "revenue" in first_reply, first_reply[:90])


# ---------------------------------------------------------------------------
# 2. Refusing when the data cannot answer
# ---------------------------------------------------------------------------

def test_refusal(store) -> None:
    print("\n2. The agent says so when the data cannot answer")

    agent, _ = make_agent(store, [
        ("text", f"That {REFUSAL_PHRASE} - there is no cost column, so profit "
                 f"cannot be worked out."),
    ])
    outcome = agent.ask("What was the profit per product?", log=False)

    check("a refusal is marked as a refusal", outcome.refused, outcome.answer)
    check("a refusal still counts as handled, not an error", outcome.success)
    check("no SQL is reported for a refusal", outcome.sql is None, str(outcome.sql))

    # The wording detector itself.
    check("detector spots the exact phrase", _looks_like_a_refusal(f"This {REFUSAL_PHRASE}."))
    check("detector spots a paraphrase",
          _looks_like_a_refusal("There is no column for supplier in the data."))
    check("detector does not fire on a normal answer",
          not _looks_like_a_refusal("South fell the most, down 3.2 million rupees."))


# ---------------------------------------------------------------------------
# 3. An answer with no query behind it is not accepted
# ---------------------------------------------------------------------------

def test_answer_must_come_from_a_query(store) -> None:
    print("\n3. An answer invented without running SQL is rejected")

    agent, _ = make_agent(store, [
        ("text", "Total revenue was about 95 million rupees."),   # never ran a query
    ])
    outcome = agent.ask("What is total revenue?", log=False)

    check("an answer with no query behind it is not reported as success",
          not outcome.success, outcome.answer)
    check("the user is told why, not given the invented number",
          "95 million" not in outcome.answer, outcome.answer)


# ---------------------------------------------------------------------------
# 4. The step cap
# ---------------------------------------------------------------------------

def test_step_cap(store) -> None:
    print("\n4. A looping agent is stopped by the step cap")

    # A model that only ever calls list_tables, for ever.
    agent, fake = make_agent(
        store,
        [("tool", "list_tables", {})] * 50,
        max_steps=4,
    )
    outcome = agent.ask("Go round in circles please", log=False)

    check("it stopped instead of looping for ever", not outcome.success)
    check("it used no more than the cap", fake.turns <= 4, f"{fake.turns} turns")
    check("the message is friendly, not a stack trace",
          "could not work this out" in outcome.answer.lower(), outcome.answer)


# ---------------------------------------------------------------------------
# 5. Charts bind to the right result
# ---------------------------------------------------------------------------

def test_chart_binding(store) -> None:
    print("\n5. A chart is drawn from the result that has the charted columns")

    agent, _ = make_agent(store, [
        ("tool", "run_sql", {"query":
            "SELECT region, SUM(quantity) AS units FROM sales GROUP BY region"}),
        # A follow-up query with completely different columns.
        ("tool", "run_sql", {"query": "SELECT COUNT(*) AS n FROM sales"}),
        # Charting the first result's columns, after the second query ran.
        ("tool", "make_chart", {"chart_type": "bar", "x": "region", "y": "units",
                                "title": "Units by region"}),
        ("text", "North sold the most units."),
    ])
    outcome = agent.ask("Units by region, and the total row count?", log=False)

    check("a chart was requested", outcome.chart_request is not None)
    check("the chart is bound to the result holding region/units",
          outcome.chart_result is not None
          and set(outcome.chart_result.column_names) == {"region", "units"},
          str(outcome.chart_result.column_names if outcome.chart_result else None))
    check("the answer still rests on the most recent query",
          outcome.result is not None and outcome.result.column_names == ["n"],
          str(outcome.result.column_names if outcome.result else None))
    check("the UI is told the chart came from a different query",
          outcome.chart_is_from_another_query)


# ---------------------------------------------------------------------------
# 6. Logging
# ---------------------------------------------------------------------------

def test_logging(store) -> None:
    print("\n6. Each question is written to the log")

    logger.clear_log()
    agent, _ = make_agent(store, [
        ("tool", "run_sql", {"query": "SELECT COUNT(*) AS n FROM sales"}),
        ("text", "There are 4,974 rows."),
    ])
    agent.ask("How many rows are there?", log=True)

    frame = logger.read_log()
    check("one row was logged", len(frame) == 1, f"got {len(frame)}")
    if len(frame):
        row = frame.iloc[0]
        check("the question was logged", row["question"] == "How many rows are there?")
        check("success was logged", bool(row["success"]))
        check("the SQL was logged", "COUNT" in str(row["sql"]).upper())
    logger.clear_log()


# ---------------------------------------------------------------------------
# Optional: ask the real model
# ---------------------------------------------------------------------------

LIVE_QUESTIONS = [
    ("sales", "Which region's sales dropped the most last quarter?", "answer"),
    ("sales", "What are the top 3 products by revenue?", "answer"),
    ("shop_orders", "Which customer segment has the highest average order value?", "answer"),
    ("shop_orders", "How many customers have never placed an order?", "answer"),
    ("sales", "What is the profit margin per product and who is our best supplier?", "refusal"),
]


def test_live() -> None:
    print("\n7. Asking the real model (needs GROQ_API_KEY)")
    for dataset, question, expectation in LIVE_QUESTIONS:
        store = load_sample_dataset(dataset)
        try:
            agent = Agent(store)
            outcome = agent.ask(question, log=False)
            if expectation == "refusal":
                check(f"[{dataset}] refused: {question[:46]}", outcome.refused, outcome.answer[:90])
            else:
                check(f"[{dataset}] answered in {outcome.duration_seconds:.0f}s, "
                      f"{outcome.self_corrections} correction(s): {question[:40]}",
                      outcome.success and not outcome.refused, outcome.answer[:90])
                print(f"        -> {outcome.answer[:140]}")
        except Exception as error:
            check(f"[{dataset}] {question[:46]}", False, f"{type(error).__name__}: {error}")
        finally:
            store.cleanup()


def main() -> int:
    live = "--live" in sys.argv

    print("=" * 74)
    print("DataPilot agent tests" + ("  (including live model calls)" if live else ""))
    print("=" * 74)

    store = load_sample_dataset("sales")
    try:
        test_self_correction(store)
        test_refusal(store)
        test_answer_must_come_from_a_query(store)
        test_step_cap(store)
        test_chart_binding(store)
        test_logging(store)
    finally:
        store.cleanup()

    if live:
        test_live()

    print("\n" + "=" * 74)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
