"""
Tests for the safety layers and the query executor (Phase 2).

Run it from the project root:

    python tests/test_safety.py

It needs no test framework -- plain asserts and printed output, so it is easy
to read and easy to demo. It checks three things:

  1. ordinary SELECT queries run and give the right answers
  2. every kind of dangerous query is refused, with a readable reason
  3. the row limit and the timeout really are enforced
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow "python tests/test_safety.py" from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_sample_dataset           # noqa: E402
from src.executor import Executor, QueryError, QueryTimeout  # noqa: E402
from src.sql_guard import UnsafeSQLError, check_sql        # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion and print it, instead of stopping at the first failure."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  {detail}")


# ---------------------------------------------------------------------------
# 1. Safe queries must work
# ---------------------------------------------------------------------------

SAFE_QUERIES = [
    # (description, sql, a function checking the result)
    (
        "simple count",
        "SELECT COUNT(*) AS n FROM sales",
        lambda r: r.dataframe["n"][0] == 4974,
    ),
    (
        "filter and aggregate",
        "SELECT SUM(quantity) AS q FROM sales WHERE region = 'South'",
        lambda r: r.dataframe["q"][0] > 0,
    ),
    (
        "group by with ordering",
        "SELECT region, COUNT(*) AS orders FROM sales GROUP BY region ORDER BY orders DESC",
        lambda r: r.row_count == 4,
    ),
    (
        "WITH ... SELECT (a CTE)",
        """
        WITH revenue AS (
            SELECT region, SUM(quantity * unit_price * (1 - discount)) AS rev
            FROM sales GROUP BY region
        )
        SELECT region, rev FROM revenue ORDER BY rev DESC
        """,
        lambda r: r.row_count == 4 and r.dataframe["rev"][0] > 0,
    ),
    (
        "date functions",
        "SELECT QUARTER(date) AS q, COUNT(*) AS n FROM sales GROUP BY q ORDER BY q",
        lambda r: r.row_count == 4,
    ),
    (
        "UNION of two selects",
        "SELECT 'a' AS label, 1 AS v UNION ALL SELECT 'b', 2",
        lambda r: r.row_count == 2,
    ),
]


def test_safe_queries() -> None:
    print("\n1. Safe queries should run")
    store = load_sample_dataset("sales")
    try:
        with Executor(store) as executor:
            for label, sql, verify in SAFE_QUERIES:
                try:
                    result = executor.run(sql)
                    check(f"{label}: ran and returned the expected shape",
                          verify(result),
                          f"got {result.row_count} rows: {result.dataframe.head(2)}")
                except Exception as error:
                    check(f"{label}: ran without error", False, f"{type(error).__name__}: {error}")
    finally:
        store.cleanup()


# ---------------------------------------------------------------------------
# 2. Dangerous queries must be refused
# ---------------------------------------------------------------------------

DANGEROUS_QUERIES = [
    ("DROP a table", "DROP TABLE sales"),
    ("DELETE rows", "DELETE FROM sales WHERE region = 'South'"),
    ("UPDATE rows", "UPDATE sales SET unit_price = 0"),
    ("INSERT rows", "INSERT INTO sales VALUES ('x')"),
    ("CREATE a new table", "CREATE TABLE stolen AS SELECT * FROM sales"),
    ("ALTER a table", "ALTER TABLE sales ADD COLUMN hacked INTEGER"),
    ("two statements at once (SQL injection style)",
     "SELECT 1; DROP TABLE sales"),
    ("read a file from the machine",
     "SELECT * FROM read_csv_auto('/etc/passwd')"),
    ("read a parquet file inside a subquery",
     "SELECT * FROM sales WHERE region IN (SELECT * FROM read_parquet('/tmp/x.parquet'))"),
    ("write results out to a file", "COPY (SELECT * FROM sales) TO '/tmp/leak.csv'"),
    ("attach another database", "ATTACH '/etc/passwd' AS leak"),
    ("list the filesystem", "SELECT * FROM glob('/**')"),
    ("change engine settings", "SET memory_limit = '99GB'"),
    ("PRAGMA to inspect the server", "PRAGMA database_list"),
    ("install an extension", "INSTALL httpfs"),
    ("query a table that does not exist", "SELECT * FROM salaries"),
    ("hide DROP behind a comment", "SELECT 1 /* nice query */ ; DROP TABLE sales"),
    ("not SQL at all", "please delete everything"),
]


def test_dangerous_queries() -> None:
    print("\n2. Dangerous queries should be refused, with a readable reason")
    store = load_sample_dataset("sales")
    try:
        with Executor(store) as executor:
            for label, sql in DANGEROUS_QUERIES:
                try:
                    executor.run(sql)
                    check(f"{label}: refused", False, "IT WAS ALLOWED TO RUN")
                except (UnsafeSQLError, QueryError) as error:
                    message = str(error)
                    readable = len(message) > 15 and "Traceback" not in message
                    check(f"{label}: refused -> {message[:68]}", readable,
                          "message is not helpful")
    finally:
        store.cleanup()


# ---------------------------------------------------------------------------
# 3. Limits must be enforced
# ---------------------------------------------------------------------------

def test_limits() -> None:
    print("\n3. Row limits and timeouts should be enforced")
    store = load_sample_dataset("sales")
    try:
        # A query with no LIMIT gets one added.
        guard = check_sql("SELECT * FROM sales", allowed_tables=["sales"], max_rows=100)
        check("a missing LIMIT is added", "LIMIT 100" in guard.safe_sql.upper(), guard.safe_sql)
        check("the change is explained to the user", bool(guard.notes), str(guard.notes))

        # An over-large LIMIT is lowered.
        guard = check_sql("SELECT * FROM sales LIMIT 999999",
                          allowed_tables=["sales"], max_rows=100)
        check("an over-large LIMIT is lowered", "LIMIT 100" in guard.safe_sql.upper(),
              guard.safe_sql)

        # A small LIMIT the user asked for is respected.
        guard = check_sql("SELECT * FROM sales LIMIT 5", allowed_tables=["sales"], max_rows=100)
        check("a small LIMIT is left alone", "LIMIT 5" in guard.safe_sql.upper(),
              guard.safe_sql)

        # The limit really caps the rows that come back.
        with Executor(store, max_rows=25) as executor:
            result = executor.run("SELECT * FROM sales")
            check("the row cap is applied to real results", result.row_count == 25,
                  f"got {result.row_count}")

        # A deliberately enormous query is cancelled rather than hanging.
        with Executor(store, timeout_seconds=2) as executor:
            started = time.perf_counter()
            try:
                executor.run("SELECT COUNT(*) FROM range(100000000000)")
                check("a runaway query is stopped", False, "it finished, unexpectedly")
            except QueryTimeout:
                elapsed = time.perf_counter() - started
                check(f"a runaway query is stopped after ~{elapsed:.1f}s", elapsed < 10,
                      f"took {elapsed:.1f}s")
    finally:
        store.cleanup()


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 74)
    print("DataPilot safety tests")
    print("=" * 74)

    test_safe_queries()
    test_dangerous_queries()
    test_limits()

    print("\n" + "=" * 74)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
