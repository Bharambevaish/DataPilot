"""
Ask DataPilot a question from the command line.

Useful for testing the agent without starting the web app.

    python scripts/ask.py "which region had the highest revenue?"
    python scripts/ask.py --dataset shop_orders "how many customers never ordered?"
    python scripts/ask.py --file mydata.csv "what is the average order value?"

It prints the answer, the SQL the agent used, the result table and the steps
it took, so you can see exactly how it reached the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Agent, AgentError                        # noqa: E402
from src.data_loader import DataLoadError, load_files, load_sample_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask DataPilot a question.")
    parser.add_argument("question", help="the question, in plain English")
    parser.add_argument("--dataset", default="sales",
                        choices=["sales", "shop_orders"],
                        help="which bundled sample dataset to use (default: sales)")
    parser.add_argument("--file", action="append", default=None,
                        help="use your own CSV/XLSX file instead of a sample "
                             "(repeat the flag for several files)")
    parser.add_argument("--no-hints", action="store_true",
                        help="leave the value hints out of the schema, to compare accuracy")
    arguments = parser.parse_args()

    # Load the data.
    try:
        if arguments.file:
            store = load_files(arguments.file)
        else:
            store = load_sample_dataset(arguments.dataset)
    except DataLoadError as error:
        print(f"Could not load the data: {error}")
        return 1

    try:
        agent = Agent(store, include_value_hints=not arguments.no_hints)
    except AgentError as error:
        print(f"Could not start the agent: {error}")
        store.cleanup()
        return 1

    try:
        outcome = agent.ask(arguments.question)

        print("=" * 76)
        print(f"Q: {outcome.question}")
        print("=" * 76)
        print(f"\n{outcome.answer}\n")

        if outcome.sql:
            print("-" * 76)
            print("SQL used:")
            print(outcome.sql)

        if outcome.dataframe is not None and not outcome.dataframe.empty:
            print("-" * 76)
            print(f"Result ({outcome.result.row_count:,} rows):")
            print(outcome.dataframe.head(15).to_string(index=False))

        if outcome.chart_request:
            print("-" * 76)
            print(f"Chart requested: {outcome.chart_request.chart_type} "
                  f"(x={outcome.chart_request.x}, y={outcome.chart_request.y})")

        print("-" * 76)
        print("How it got there:")
        for number, line in enumerate(outcome.step_summary(), start=1):
            print(f"  {number}. {line}")

        print("-" * 76)
        print(f"success={outcome.success}  refused={outcome.refused}  "
              f"llm_turns={outcome.llm_turns}  self_corrections={outcome.self_corrections}  "
              f"time={outcome.duration_seconds:.1f}s  model={outcome.model}")

        return 0 if outcome.success else 1
    finally:
        store.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
