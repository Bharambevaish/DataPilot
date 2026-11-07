"""
Measure how good the agent actually is.

    python eval/evaluate.py                       # run all 30 questions
    python eval/evaluate.py --label v2            # name this run
    python eval/evaluate.py --no-hints            # schema without value hints
    python eval/evaluate.py --only h04,m02        # re-run specific questions

WHY COMPARE RESULTS AND NOT SQL TEXT
------------------------------------
The obvious way to score a text-to-SQL system is to compare its query with the
"correct" query. That is a bad measure. These two are the same answer:

    SELECT region, SUM(x) FROM t GROUP BY region ORDER BY 2 DESC LIMIT 1
    SELECT region, SUM(x) AS r FROM t GROUP BY 1 HAVING r = (SELECT MAX(...))

and string comparison marks the second wrong. So we run both queries and
compare what comes back. That is **execution accuracy**, and it is the
standard measure in the text-to-SQL literature.

TWO SCORES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
---------------------------------------------------
1. Execution accuracy (strict)
   The agent's result table matches the gold table: same number of rows, and
   every gold column has a matching column of values. Row order and column
   names are ignored, numbers are rounded, because none of those change the
   answer.

2. Answer accuracy (what the user sees)
   The agent's sentence contains the gold figure. This exists because of a
   real pattern: asked "which region fell the most", the agent often returns
   all four regions and *says* "South". Strict execution accuracy calls that a
   miss because the table has 4 rows instead of 1 -- but the person reading
   the app got the right answer. Reporting both, and the gap between them, is
   more honest than picking whichever number flatters the project.

For the three unanswerable questions, correct means refusing. Producing a
confident number there is the worst failure mode a data tool can have, so it
is scored separately.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd   # noqa: E402

from src.agent import Agent, AgentError            # noqa: E402
from src.data_loader import load_sample_dataset    # noqa: E402
from src.executor import Executor                  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"
DIFFICULTY_ORDER = ["easy", "medium", "hard", "unanswerable"]


# ---------------------------------------------------------------------------
# Comparing results
# ---------------------------------------------------------------------------

# Models write typographically: a non-breaking hyphen in "Cotton T-Shirt", a
# narrow space inside a number. Those are the same characters to a reader, so
# they are folded to plain ASCII before anything is compared. Found by the
# first evaluation run, which scored a correct answer as wrong for exactly
# this reason.
_DASHES = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_SPACES = dict.fromkeys(
    map(ord, "\u00a0\u2007\u2009\u200a\u202f\u2060"), " ")


def plain_text(value: Any) -> str:
    """Fold fancy punctuation to ASCII, collapse spaces, lower-case."""
    text = str(value).translate(_DASHES).translate(_SPACES)
    return re.sub(r"\s+", " ", text).strip().lower()


def normalise(value: Any) -> str:
    """
    Reduce one cell to a comparable string.

    Numbers are rounded to 2 decimal places, so 45600.2300001 and 45600.23 are
    the same answer. Dates lose a midnight time part. Text is trimmed and
    lower-cased, so 'South' and 'south ' match.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):          # catches NaN, NaT and pandas' NA
            return ""
    except (TypeError, ValueError):
        pass                        # arrays and lists are not missing values
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{round(float(value), 2):.2f}"
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        if stamp.hour == stamp.minute == stamp.second == 0:
            return stamp.strftime("%Y-%m-%d")
        return stamp.strftime("%Y-%m-%d %H:%M:%S")
    return plain_text(value)


def column_values(series: pd.Series) -> Counter:
    """A column as a bag of normalised values -- so row order does not matter."""
    return Counter(normalise(v) for v in series.tolist())


def results_match(gold: pd.DataFrame, got: pd.DataFrame | None) -> bool:
    """
    Execution accuracy: does the agent's table say the same thing as gold's?

    Column *names* are ignored (the agent may call it `revenue` or `total`),
    and so is row order. What must match is the number of rows and, for every
    gold column, the bag of values in some column of the agent's result.
    """
    if got is None or gold is None:
        return False
    if len(gold) != len(got):
        return False
    if gold.empty and got.empty:
        return True

    unused = list(got.columns)
    for gold_column in gold.columns:
        wanted = column_values(gold[gold_column])
        for candidate in unused:
            if column_values(got[candidate]) == wanted:
                unused.remove(candidate)   # one agent column per gold column
                break
        else:
            return False
    return True


# -- the "did the sentence say it" check ------------------------------------

_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6,
           "bn": 1e9, "billion": 1e9, "cr": 1e7, "crore": 1e7, "lakh": 1e5}

_NUMBER_PATTERN = re.compile(
    r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(k|thousand|mn|m|million|bn|billion|cr|crore|lakh)?\b",
    re.IGNORECASE,
)


def numbers_in(text: str) -> list[float]:
    """
    Pull every number out of a sentence, understanding scale words.

    "about $3.20 million" -> 3200000.0, "₹45,600" -> 45600.0, "4,974" -> 4974.0
    """
    found: list[float] = []
    for digits, scale in _NUMBER_PATTERN.findall(text or ""):
        try:
            number = float(digits.replace(",", ""))
        except ValueError:
            continue
        if scale:
            number *= _SCALES[scale.lower()]
        found.append(number)
    return found


def number_is_present(target: float, candidates: list[float]) -> bool:
    """
    Is this figure in the sentence?

    A whole number must appear as itself -- a count of 32 is not 33. Anything
    with decimals is allowed 1% slack, because the agent legitimately writes
    "about 3.20 million" for 3,199,509.23 and "down 38.5%" for -38.47.
    Absolute values count: "fell by 38.5%" is the right answer for -38.47.
    """
    targets = {target, abs(target)}
    for wanted in targets:
        whole = float(wanted).is_integer()
        tolerance = 0.5 if whole else max(0.01, abs(wanted) * 0.01)
        for candidate in candidates:
            if abs(candidate - wanted) <= tolerance or abs(abs(candidate) - wanted) <= tolerance:
                return True
    return False


def answer_states_gold(gold: pd.DataFrame, answer: str) -> bool:
    """
    Does the agent's sentence contain the headline figure from the gold result?

    The gold queries are ordered, so the first row is the headline. For a
    multi-row gold result we only require that first row -- nobody writes out
    twelve months in a sentence, and the table underneath shows the rest.
    """
    if gold is None or gold.empty or not answer:
        return False

    text = str(answer)
    candidates = numbers_in(plain_text(text))
    lowered = plain_text(text)

    for value in gold.iloc[0].tolist():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if pd.isna(value):
                continue
            if not number_is_present(float(value), candidates):
                return False
        else:
            as_text = normalise(value)
            if not as_text:
                continue
            # A date may legitimately be written as "December 2024".
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_text):
                stamp = pd.Timestamp(as_text)
                spellings = [as_text, stamp.strftime("%B %Y").lower(),
                             stamp.strftime("%b %Y").lower(),
                             stamp.strftime("%B").lower()]
                if not any(s in lowered for s in spellings):
                    return False
            elif as_text not in lowered:
                return False
    return True


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    id: str
    difficulty: str
    dataset: str
    question: str
    correct: bool = False              # the headline score for this question
    execution_match: bool = False
    answer_match: bool = False
    refused: bool = False
    expected_refusal: bool = False
    agent_sql: str | None = None
    agent_answer: str = ""
    gold_rows: int | None = None
    agent_rows: int | None = None
    self_corrections: int = 0
    llm_turns: int = 0
    duration_seconds: float = 0.0
    failure_reason: str = ""


def evaluate_one(
    question: dict[str, Any],
    agent: Agent,
    gold_executor: Executor,
) -> QuestionResult:
    """Ask the agent one question and score the answer."""
    outcome_record = QuestionResult(
        id=question["id"],
        difficulty=question["difficulty"],
        dataset=question["dataset"],
        question=question["question"],
        expected_refusal=question.get("gold_sql") is None,
    )

    try:
        outcome = agent.ask(question["question"], log=False)
    except Exception as error:                      # noqa: BLE001
        outcome_record.failure_reason = f"the agent crashed: {type(error).__name__}: {error}"
        return outcome_record

    outcome_record.agent_answer = outcome.answer
    outcome_record.agent_sql = outcome.sql
    outcome_record.refused = outcome.refused
    outcome_record.self_corrections = outcome.self_corrections
    outcome_record.llm_turns = outcome.llm_turns
    outcome_record.duration_seconds = outcome.duration_seconds
    if outcome.result is not None:
        outcome_record.agent_rows = outcome.result.row_count

    # --- unanswerable: the only correct behaviour is to refuse --------------
    if outcome_record.expected_refusal:
        outcome_record.correct = bool(outcome.refused)
        if not outcome_record.correct:
            outcome_record.failure_reason = (
                "answered a question the data cannot support, instead of refusing"
            )
        return outcome_record

    # --- answerable: compare against the gold result ------------------------
    try:
        gold = gold_executor.run(question["gold_sql"]).dataframe
        outcome_record.gold_rows = len(gold)
    except Exception as error:                      # noqa: BLE001
        outcome_record.failure_reason = f"the gold SQL itself failed: {error}"
        return outcome_record

    if not outcome.success:
        outcome_record.failure_reason = f"no answer produced: {outcome.answer[:120]}"
        return outcome_record

    if outcome.refused:
        outcome_record.failure_reason = "refused a question the data can actually answer"
        return outcome_record

    got = outcome.result.dataframe if outcome.result is not None else None
    outcome_record.execution_match = results_match(gold, got)
    outcome_record.answer_match = answer_states_gold(gold, outcome.answer)

    # The headline score: either the table matched, or the sentence stated the
    # right figure. Both mean the person got a correct answer.
    outcome_record.correct = outcome_record.execution_match or outcome_record.answer_match

    if not outcome_record.correct:
        if got is None:
            outcome_record.failure_reason = "no result table was produced"
        elif outcome_record.agent_rows != outcome_record.gold_rows:
            outcome_record.failure_reason = (
                f"wrong values and wrong shape "
                f"(gold {outcome_record.gold_rows} rows, agent {outcome_record.agent_rows})"
            )
        else:
            outcome_record.failure_reason = "same shape, different numbers"
    elif not outcome_record.execution_match:
        outcome_record.failure_reason = (
            "answer correct, table shaped differently from gold "
            f"(gold {outcome_record.gold_rows} rows, agent {outcome_record.agent_rows})"
        )

    return outcome_record


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    label: str
    model: str
    value_hints: bool
    when: str
    results: list[QuestionResult] = field(default_factory=list)

    def answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.expected_refusal]

    def unanswerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.expected_refusal]

    def accuracy(self, subset: list[QuestionResult] | None = None) -> float:
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        return round(100.0 * sum(r.correct for r in rows) / len(rows), 1)

    def by_difficulty(self) -> dict[str, tuple[int, int, float]]:
        out: dict[str, tuple[int, int, float]] = {}
        for level in DIFFICULTY_ORDER:
            rows = [r for r in self.results if r.difficulty == level]
            if rows:
                correct = sum(r.correct for r in rows)
                out[level] = (correct, len(rows), round(100.0 * correct / len(rows), 1))
        return out

    def stats(self) -> dict[str, Any]:
        rows = self.results
        answerable = self.answerable()
        return {
            "total": len(rows),
            "correct": sum(r.correct for r in rows),
            "accuracy": self.accuracy(),
            "execution_accuracy": round(
                100.0 * sum(r.execution_match for r in answerable) / max(len(answerable), 1), 1),
            "answer_accuracy": round(
                100.0 * sum(r.answer_match for r in answerable) / max(len(answerable), 1), 1),
            "refusal_accuracy": self.accuracy(self.unanswerable()),
            "avg_seconds": round(sum(r.duration_seconds for r in rows) / max(len(rows), 1), 1),
            "total_self_corrections": sum(r.self_corrections for r in rows),
            "questions_needing_a_correction": sum(1 for r in rows if r.self_corrections > 0),
            "self_correction_rate": round(
                100.0 * sum(1 for r in rows if r.self_corrections > 0) / max(len(rows), 1), 1),
            "avg_llm_turns": round(sum(r.llm_turns for r in rows) / max(len(rows), 1), 2),
        }


def run_evaluation(
    questions: list[dict[str, Any]],
    label: str,
    value_hints: bool = True,
    workers: int = 2,
) -> RunSummary:
    """Run every question and collect the scores."""
    datasets = sorted({q["dataset"] for q in questions})
    stores = {name: load_sample_dataset(name) for name in datasets}

    try:
        agents: dict[str, Agent] = {}
        for name, store in stores.items():
            agents[name] = Agent(store, include_value_hints=value_hints)

        summary = RunSummary(
            label=label,
            model=next(iter(agents.values())).model,
            value_hints=value_hints,
            when=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        print(f"\nRunning '{label}'  ({len(questions)} questions, "
              f"value hints {'on' if value_hints else 'OFF'}, {workers} at a time)")
        print("-" * 78)

        def task(question: dict[str, Any]) -> QuestionResult:
            # A fresh Executor per question keeps the gold queries off the
            # agent's connection, so parallel runs cannot interfere.
            with Executor(stores[question["dataset"]]) as gold_executor:
                return evaluate_one(question, agents[question["dataset"]], gold_executor)

        def describe(result: QuestionResult, done: int) -> str:
            mark = "ok  " if result.correct else "FAIL"
            extra = ""
            if result.correct and not result.execution_match and not result.expected_refusal:
                extra = "  (right answer, different table shape)"
            elif not result.correct:
                extra = f"  <- {result.failure_reason}"
            return (f"  [{done:2}/{len(questions)}] {mark}  {result.id}  "
                    f"{result.difficulty:12} {result.question[:44]:46}{extra}")

        # Report each question as it finishes rather than at the end: a full
        # run takes several minutes, and silence looks like a hang.
        started = time.perf_counter()
        results: list[QuestionResult] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, question) for question in questions]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(describe(result, len(results)), flush=True)

        # Put them back into the original question order for the report.
        order = {q["id"]: i for i, q in enumerate(questions)}
        summary.results = sorted(results, key=lambda r: order[r.id])

        print("-" * 78)
        print(f"wall clock: {time.perf_counter() - started:.0f}s")
        return summary
    finally:
        for store in stores.values():
            store.cleanup()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_run(summary: RunSummary) -> Path:
    """Save the raw scores, so before/after comparisons can be rebuilt later."""
    path = EVAL_DIR / f"run_{summary.label}.json"
    path.write_text(json.dumps({
        "label": summary.label,
        "model": summary.model,
        "value_hints": summary.value_hints,
        "when": summary.when,
        "stats": summary.stats(),
        "results": [asdict(r) for r in summary.results],
    }, indent=2, default=str), encoding="utf-8")
    return path


def print_summary(summary: RunSummary) -> None:
    stats = summary.stats()
    print(f"\n=== {summary.label} ===")
    print(f"  overall accuracy        {stats['accuracy']}%  "
          f"({stats['correct']}/{stats['total']})")
    for level, (correct, total, percent) in summary.by_difficulty().items():
        print(f"    {level:14}        {percent}%  ({correct}/{total})")
    print(f"  execution accuracy      {stats['execution_accuracy']}%  (strict table match)")
    print(f"  answer accuracy         {stats['answer_accuracy']}%  (figure stated in the sentence)")
    print(f"  refusals correct        {stats['refusal_accuracy']}%")
    print(f"  self-correction rate    {stats['self_correction_rate']}%  "
          f"({stats['total_self_corrections']} corrections in total)")
    print(f"  average time            {stats['avg_seconds']}s per question")
    print(f"  average model turns     {stats['avg_llm_turns']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the DataPilot agent.")
    parser.add_argument("--label", default="run", help="a name for this run")
    parser.add_argument("--no-hints", action="store_true",
                        help="remove value hints from the schema, to measure their effect")
    parser.add_argument("--only", default=None,
                        help="comma-separated question ids, e.g. h04,m02")
    parser.add_argument("--difficulty", default=None,
                        help="only run one difficulty level")
    parser.add_argument("--workers", type=int, default=2,
                        help="how many questions to run at once (default 2; Groq's free tier is token-per-minute limited)")
    arguments = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]

    if arguments.only:
        wanted = {q.strip() for q in arguments.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if arguments.difficulty:
        questions = [q for q in questions if q["difficulty"] == arguments.difficulty]
    if not questions:
        print("No questions matched those filters.")
        return 1

    try:
        summary = run_evaluation(
            questions, label=arguments.label,
            value_hints=not arguments.no_hints, workers=arguments.workers,
        )
    except AgentError as error:
        print(f"Could not start the agent: {error}")
        return 1

    print_summary(summary)
    path = save_run(summary)
    print(f"\nRaw scores written to {path.relative_to(EVAL_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
