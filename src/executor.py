"""
Run SQL against the loaded data, safely.

This file holds safety layers 2 and 3 (layer 1 is src/sql_guard.py):

Layer 2 -- the connection itself cannot do damage:
    * it is opened read_only=True, so DuckDB refuses every statement that
      would change the database, including CREATE, INSERT, UPDATE and DELETE
    * it is opened with enable_external_access=False, so the engine refuses to
      touch the filesystem or the network at all. That blocks COPY ... TO,
      read_csv('/etc/passwd'), ATTACH and INSTALL, even if such a query
      somehow got past the parser.

Layer 3 -- no query can run away with the machine:
    * every query carries a row limit (added by sql_guard)
    * every query has a wall-clock timeout; when it is reached we call
      DuckDB's interrupt() and report a friendly timeout message.

Why this matters: layer 2 was tested on its own and blocks all of the
dangerous statements by itself. So a bug in my parser code is not enough to
lose data -- an attacker would have to defeat the parser *and* the read-only
file handle *and* the engine's file-access switch.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from src import config
from src.data_loader import DataStore
from src.sql_guard import UnsafeSQLError, check_sql

# Settings handed to DuckDB when the read-only connection is opened.
READ_ONLY_CONFIG = {
    # Refuse all filesystem and network access at the engine level.
    "enable_external_access": False,
    # Keep memory use sane on a small laptop.
    "memory_limit": "1GB",
    "threads": 2,
}


class QueryError(Exception):
    """A query could not be run. The message is safe to show a user."""


class QueryTimeout(QueryError):
    """A query took longer than the allowed time and was cancelled."""


@dataclass
class QueryResult:
    """The outcome of running one query."""
    dataframe: pd.DataFrame
    sql: str                              # the SQL actually run (after the guard)
    requested_sql: str                    # what the agent originally asked for
    duration_seconds: float
    notes: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.dataframe)

    @property
    def column_names(self) -> list[str]:
        return [str(c) for c in self.dataframe.columns]

    @property
    def is_empty(self) -> bool:
        return self.dataframe.empty

    def to_markdown(self, max_rows: int = 20) -> str:
        """
        Render the result as a small text table.

        This is what gets sent back to the LLM, so it is deliberately capped:
        a 1,000-row result would fill the context window and teach the model
        nothing extra.
        """
        if self.dataframe.empty:
            return "(the query ran, but returned no rows)"

        shown = self.dataframe.head(max_rows)
        lines = [" | ".join(self.column_names)]
        lines.append(" | ".join("---" for _ in self.column_names))
        for _, row in shown.iterrows():
            lines.append(" | ".join("" if pd.isna(v) else str(v) for v in row))

        text = "\n".join(lines)
        if self.row_count > max_rows:
            text += f"\n... ({self.row_count:,} rows in total, first {max_rows} shown)"
        return text


class Executor:
    """
    Runs checked SQL against one DataStore.

    Create one of these per loaded dataset and keep it; opening the DuckDB
    file is cheap but not free. Queries run one at a time, which is all the
    app and the evaluation ever need.
    """

    def __init__(
        self,
        store: DataStore,
        max_rows: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.store = store
        self.max_rows = max_rows if max_rows is not None else config.max_rows()
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None
            else config.query_timeout_seconds()
        )
        self._connection: duckdb.DuckDBPyConnection | None = None

    # -- connection handling -------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open the read-only connection, reusing it on later calls."""
        if self._connection is None:
            path = Path(self.store.db_path)
            if not path.exists():
                raise QueryError(
                    "The dataset is no longer available. Please load the data again."
                )
            try:
                self._connection = duckdb.connect(
                    str(path), read_only=True, config=READ_ONLY_CONFIG
                )
            except Exception as error:
                raise QueryError(f"Could not open the dataset: {error}") from error
        return self._connection

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    # -- running queries -----------------------------------------------------

    def table_names(self) -> list[str]:
        return self.store.table_names()

    def run(self, sql: str) -> QueryResult:
        """
        Check a query and run it.

        Raises UnsafeSQLError if the guard refuses it, QueryTimeout if it runs
        too long, or QueryError for anything else. All three carry messages
        meant to be read by a human -- and by the agent, which uses them to
        correct itself.
        """
        guard = check_sql(sql, allowed_tables=self.table_names(), max_rows=self.max_rows)
        connection = self._connect()

        outcome: dict[str, object] = {}

        def work() -> None:
            try:
                outcome["frame"] = connection.execute(guard.safe_sql).df()
            except Exception as error:   # noqa: BLE001 - reported to the caller
                outcome["error"] = error

        started = time.perf_counter()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(self.timeout_seconds)

        if worker.is_alive():
            # Ask DuckDB to stop, then wait briefly for the thread to unwind.
            try:
                connection.interrupt()
            except Exception:
                pass
            worker.join(5)
            raise QueryTimeout(
                f"The query took longer than {self.timeout_seconds} seconds and was "
                "stopped. Try narrowing it down, for example by filtering to fewer "
                "rows or grouping into fewer groups."
            )

        duration = time.perf_counter() - started

        if "error" in outcome:
            error = outcome["error"]
            raise QueryError(_readable_duckdb_error(error)) from error  # type: ignore[arg-type]

        frame = outcome.get("frame")
        if frame is None:
            raise QueryError("The query returned nothing at all, which should not happen.")

        return QueryResult(
            dataframe=frame,                      # type: ignore[arg-type]
            sql=guard.safe_sql,
            requested_sql=guard.original_sql,
            duration_seconds=duration,
            notes=guard.notes,
            tables_used=guard.tables_used,
        )

    # Allow "with Executor(store) as ex:" so connections always get closed.
    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exception_details: object) -> None:
        self.close()


def _readable_duckdb_error(error: Exception) -> str:
    """
    Turn a DuckDB exception into something worth reading.

    DuckDB messages are good but verbose, and they include the whole query
    plus an arrow diagram. We keep the first line, which is the actual
    problem, and add a nudge when the cause is a typo in a column name.
    """
    text = str(error).strip()
    first_line = text.split("\n", 1)[0].strip()

    # Strip DuckDB's "Binder Error: " style prefix, which means nothing to a user.
    for prefix in ("Binder Error: ", "Catalog Error: ", "Parser Error: ",
                   "Conversion Error: ", "Invalid Input Error: "):
        if first_line.startswith(prefix):
            first_line = first_line[len(prefix):]
            break

    lowered = text.lower()
    if "referenced column" in lowered or "does not have a column" in lowered:
        first_line += (
            "  (Check the column names in the schema above -- use them exactly.)"
        )
    elif "read-only" in lowered or "read only" in lowered:
        first_line += "  (DataPilot is read-only: data can be read but never changed.)"

    return first_line


def run_sql(store: DataStore, sql: str) -> QueryResult:
    """Convenience helper for one-off queries in scripts and tests."""
    with Executor(store) as executor:
        return executor.run(sql)


__all__ = [
    "Executor", "QueryResult", "QueryError", "QueryTimeout",
    "UnsafeSQLError", "run_sql",
]
