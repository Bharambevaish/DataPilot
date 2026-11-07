"""
Write one line per question to logs/queries.jsonl.

Why bother? Because "it works on the five questions I tried" is not evidence.
A log turns the app into something measurable: how often the agent succeeds,
how often it has to correct itself, how long it takes, and which questions it
struggles with. The Streamlit "Query logs" tab reads this file, and the
failures are what I used to improve the prompt in Phase 5.

Format: JSON Lines -- one JSON object per line. Appending is cheap, a crash
cannot corrupt earlier lines, and pandas can read the whole file in one call.

Two deliberate choices:
  * Logging never raises. A failed write must not break someone's question.
  * The log stays on the machine that runs the app; logs/*.jsonl is in
    .gitignore, so questions are never committed to the repository.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

LOG_FILE = config.LOGS_DIR / "queries.jsonl"


def log_question(
    question: str,
    answer: str = "",
    sql: str | None = None,
    success: bool = False,
    duration_seconds: float = 0.0,
    steps: int = 0,
    self_corrections: int = 0,
    tables: str = "",
    model: str = "",
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one record. Silently gives up if the file cannot be written."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "sql": sql,
        "success": bool(success),
        "duration_seconds": round(float(duration_seconds), 3),
        "steps": int(steps),
        "self_corrections": int(self_corrections),
        "tables": tables,
        "model": model or config.model_name(),
        "error": error,
    }
    if extra:
        record.update(extra)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Logging is useful, but never more important than answering the question.
        pass


def read_log(path: Path | None = None) -> pd.DataFrame:
    """
    Read the log into a DataFrame. Returns an empty frame if there is no log.

    Lines that are not valid JSON are skipped rather than killing the read --
    a half-written final line (say, the power went out) should not hide the
    other 500 good records.
    """
    log_path = path or LOG_FILE
    if not log_path.exists():
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def summarise(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Headline numbers for the 'Query logs' tab."""
    if frame is None:
        frame = read_log()

    if frame is None or frame.empty:
        return {
            "total_questions": 0, "successful": 0, "success_rate": 0.0,
            "avg_duration_seconds": 0.0, "avg_self_corrections": 0.0,
            "questions_with_a_correction": 0,
        }

    total = len(frame)
    successful = int(frame.get("success", pd.Series(dtype=bool)).fillna(False).sum())
    corrections = frame.get("self_corrections", pd.Series([0] * total)).fillna(0)

    return {
        "total_questions": total,
        "successful": successful,
        "success_rate": round(100.0 * successful / total, 1),
        "avg_duration_seconds": round(
            float(frame.get("duration_seconds", pd.Series([0.0])).fillna(0).mean()), 2
        ),
        "avg_self_corrections": round(float(corrections.mean()), 2),
        "questions_with_a_correction": int((corrections > 0).sum()),
    }


def clear_log() -> None:
    """Delete the log file. Used by tests so they start from a clean slate."""
    try:
        LOG_FILE.unlink(missing_ok=True)
    except Exception:
        pass
