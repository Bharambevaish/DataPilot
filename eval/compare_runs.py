"""
Print a side-by-side table of saved evaluation runs.

    python eval/compare_runs.py                    # every run_*.json
    python eval/compare_runs.py v1-baseline v2-improved

Each run of evaluate.py writes eval/run_<label>.json. This reads those files
and prints a markdown table, so the comparison in results.md is copied from
real numbers rather than typed out by hand (and re-typed slightly wrong).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

ROWS = [
    ("Overall accuracy",            "accuracy",                      "%"),
    ("Execution accuracy (strict)", "execution_accuracy",            "%"),
    ("Answer accuracy",             "answer_accuracy",               "%"),
    ("Unanswerable refused",        "refusal_accuracy",              "%"),
    ("Self-correction rate",        "self_correction_rate",          "%"),
    ("Average seconds / question",  "avg_seconds",                   "s"),
    ("Average model turns",         "avg_llm_turns",                 ""),
]


def load_runs(labels: list[str]) -> list[dict]:
    if labels:
        paths = [EVAL_DIR / f"run_{label}.json" for label in labels]
    else:
        paths = sorted(EVAL_DIR.glob("run_*.json"))

    runs = []
    for path in paths:
        if not path.exists():
            print(f"(skipping {path.name}: not found)")
            continue
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def main() -> int:
    runs = load_runs(sys.argv[1:])
    if not runs:
        print("No saved runs. Run: python eval/evaluate.py --label myrun")
        return 1

    names = [r["label"] for r in runs]
    width = max(len(n) for n in names + ["Measure"]) + 2

    print()
    print("| " + "Measure".ljust(28) + " | " + " | ".join(n.ljust(width) for n in names) + " |")
    print("|" + "-" * 30 + "|" + "|".join("-" * (width + 2) for _ in names) + "|")

    for title, key, unit in ROWS:
        cells = [f"{r['stats'].get(key, '?')}{unit}".ljust(width) for r in runs]
        print(f"| {title.ljust(28)} | " + " | ".join(cells) + " |")

    # Accuracy split by difficulty.
    print()
    levels = ["easy", "medium", "hard", "unanswerable"]
    print("| " + "By difficulty".ljust(28) + " | " + " | ".join(n.ljust(width) for n in names) + " |")
    print("|" + "-" * 30 + "|" + "|".join("-" * (width + 2) for _ in names) + "|")
    for level in levels:
        cells = []
        for run in runs:
            rows = [r for r in run["results"] if r["difficulty"] == level]
            if rows:
                correct = sum(r["correct"] for r in rows)
                cells.append(
                    f"{round(100 * correct / len(rows), 1)}% ({correct}/{len(rows)})".ljust(width))
            else:
                cells.append("-".ljust(width))
        print(f"| {level.ljust(28)} | " + " | ".join(cells) + " |")

    # Which questions each run got wrong.
    print("\nFailures per run:")
    for run in runs:
        wrong = [r for r in run["results"] if not r["correct"]]
        if not wrong:
            print(f"  {run['label']}: none")
            continue
        print(f"  {run['label']}: {len(wrong)}")
        for r in wrong:
            print(f"    - {r['id']} ({r['difficulty']}): {r['failure_reason'][:92]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
