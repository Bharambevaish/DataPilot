"""
Check that generated SQL is safe to run.

This is safety layer 1 of 3. The others are:
  2. the DuckDB connection is opened read-only, and with the engine's own
     file access switched off  (src/executor.py)
  3. every query gets a row limit and a time limit  (src/executor.py)

Any one of them would stop most attacks. Together they mean a bad query has
to get past a parser, a read-only file handle and an engine-level file-access
switch. That is what "defence in depth" means, and it is the thing worth
explaining in an interview.

The important design decision here is that this is an **allowlist**, not a
blocklist. We do not search the query text for the word "DROP" -- that is easy
to fool, for example with a comment or odd spacing. Instead we parse the SQL
into a syntax tree with sqlglot and insist the whole statement is a query
(SELECT, or WITH ... SELECT, or a UNION of SELECTs). Anything that is not a
query -- INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH, COPY, PRAGMA,
SET, INSTALL, LOAD, CALL -- fails that single check automatically, including
statements nobody thought to add to a list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

DIALECT = "duckdb"

# DuckDB functions that reach outside the database, either to read files from
# the machine or to fetch things over the network. A read-only connection does
# not stop these, so they are blocked by name here as well.
FORBIDDEN_FUNCTIONS = {
    # reading files
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_json_objects", "read_ndjson", "read_ndjson_auto", "read_ndjson_objects",
    "read_text", "read_blob", "read_xlsx", "sniff_csv",
    "parquet_scan", "csv_scan", "json_scan", "parquet_metadata",
    "parquet_schema", "arrow_scan", "iceberg_scan", "delta_scan",
    # listing the filesystem
    "glob",
    # writing or fetching
    "copy_to", "write_parquet", "write_csv",
    # database internals that leak file paths
    "duckdb_databases", "duckdb_settings", "duckdb_extensions",
    "pragma_database_list", "pragma_database_size",
    # shell-ish helpers from extensions
    "shellfs", "python_scan",
}

# Any function name that starts with one of these is refused too, so a new
# DuckDB release adding read_avro() does not quietly open a hole.
FORBIDDEN_FUNCTION_PREFIXES = ("read_", "write_", "pragma_", "duckdb_")

# Friendly names for the statement types we refuse, used in error messages.
STATEMENT_LABELS = {
    exp.Insert: "INSERT", exp.Update: "UPDATE", exp.Delete: "DELETE",
    exp.Drop: "DROP", exp.Create: "CREATE", exp.Alter: "ALTER",
    exp.Attach: "ATTACH", exp.Detach: "DETACH", exp.Copy: "COPY",
    exp.Pragma: "PRAGMA", exp.Set: "SET", exp.Install: "INSTALL",
    exp.Merge: "MERGE", exp.TruncateTable: "TRUNCATE",
}


class UnsafeSQLError(Exception):
    """Raised when a query is refused. The message explains why, in plain words."""


@dataclass
class GuardResult:
    """What the guard decided."""
    safe_sql: str                             # the query to actually run
    original_sql: str
    notes: list[str] = field(default_factory=list)  # changes we made, e.g. "added LIMIT"
    tables_used: list[str] = field(default_factory=list)


def _statement_label(statement: exp.Expression) -> str:
    """A readable name for a statement we are about to refuse."""
    for node_type, label in STATEMENT_LABELS.items():
        if isinstance(statement, node_type):
            return label
    # sqlglot parses anything it does not recognise as a Command, and keeps the
    # leading keyword, which is exactly what we want to name in the error.
    if isinstance(statement, exp.Command):
        keyword = str(statement.this).strip().upper()
        return keyword or "that"
    return type(statement).__name__.upper()


def _check_no_forbidden_functions(statement: exp.Expression) -> None:
    """Refuse any call to a function that touches files or the network."""
    for node in statement.walk():
        # An unknown function (anything DuckDB-specific) parses as Anonymous.
        if isinstance(node, exp.Anonymous):
            name = str(node.this).lower()
        elif isinstance(node, exp.Func):
            name = node.sql_name().lower()
        else:
            continue

        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            raise UnsafeSQLError(
                f"The function {name}() is not allowed, because it can read files "
                "or reach the network. Only the tables already loaded can be queried."
            )


def _collect_cte_names(statement: exp.Expression) -> set[str]:
    """Names defined inside the query itself by WITH ... AS (...)."""
    names = set()
    for cte in statement.find_all(exp.CTE):
        if cte.alias:
            names.add(cte.alias.lower())
    return names


def _check_tables(statement: exp.Expression, allowed_tables: list[str] | None) -> list[str]:
    """
    Make sure the query only names tables that are actually loaded.

    This catches the most common agent mistake -- inventing a table name --
    and turns it into a clear message the agent can learn from, instead of a
    confusing DuckDB error.
    """
    cte_names = _collect_cte_names(statement)
    used: list[str] = []

    for table in statement.find_all(exp.Table):
        # A table whose "this" is a function call, e.g. read_csv('x'), is
        # handled by _check_no_forbidden_functions, so skip it here.
        if isinstance(table.this, exp.Anonymous) or isinstance(table.this, exp.Func):
            continue

        name = table.name
        if not name:
            continue
        if name.lower() in cte_names:
            continue  # defined by the query itself, not a real table

        if name not in used:
            used.append(name)

        if allowed_tables is not None:
            known = {t.lower() for t in allowed_tables}
            if name.lower() not in known:
                available = ", ".join(allowed_tables) or "none"
                raise UnsafeSQLError(
                    f'There is no table called "{name}". '
                    f"The tables you can use are: {available}."
                )

    return used


def _apply_row_limit(statement: exp.Expression, max_rows: int) -> tuple[exp.Expression, list[str]]:
    """
    Make sure the query cannot return an unbounded number of rows.

    Two cases:
      * no LIMIT at all   -> add LIMIT max_rows
      * LIMIT bigger than max_rows -> lower it to max_rows

    Lowering an over-large limit is a deliberate choice: the point of the cap
    is to protect the browser and memory, so a query asking for 10 million
    rows must not get them just because it said so out loud.
    """
    notes: list[str] = []
    existing = statement.args.get("limit")

    if existing is None:
        statement = statement.limit(max_rows)
        notes.append(f"Added LIMIT {max_rows} (no limit was given).")
        return statement, notes

    # Read the existing limit if it is a plain number.
    try:
        current = int(existing.expression.this)
    except (AttributeError, TypeError, ValueError):
        return statement, notes  # e.g. a parameter; leave it alone

    if current > max_rows:
        statement = statement.limit(max_rows)
        notes.append(f"Lowered LIMIT {current:,} to {max_rows} to keep the result manageable.")

    return statement, notes


def check_sql(
    sql: str,
    allowed_tables: list[str] | None = None,
    max_rows: int = 1000,
) -> GuardResult:
    """
    Check one SQL string and return the safe version of it.

    Raises UnsafeSQLError with a readable reason if the query is refused.
    That message is shown to the user *and* fed back to the agent, which is
    how the agent fixes its own mistakes.
    """
    if sql is None or not str(sql).strip():
        raise UnsafeSQLError("The query was empty.")

    original = str(sql).strip().rstrip(";").strip()

    # 1. It has to parse.
    try:
        statements = sqlglot.parse(original, dialect=DIALECT)
    except Exception as error:
        raise UnsafeSQLError(f"This is not valid SQL: {error}") from error

    statements = [s for s in statements if s is not None]
    if not statements:
        raise UnsafeSQLError("The query was empty.")

    # 2. Exactly one statement. This is what stops "SELECT 1; DROP TABLE x".
    if len(statements) > 1:
        raise UnsafeSQLError(
            f"Only one statement is allowed, but {len(statements)} were given. "
            "Send a single SELECT query."
        )

    statement = statements[0]

    # 3. The allowlist: the statement must be a query and nothing else.
    if not isinstance(statement, exp.Query):
        label = _statement_label(statement)
        raise UnsafeSQLError(
            f"{label} is not allowed. DataPilot is read-only, so only "
            "SELECT queries (optionally starting with WITH) can be run."
        )

    # 4. No functions that read files or reach the network.
    _check_no_forbidden_functions(statement)

    # 5. Only tables that really exist.
    tables_used = _check_tables(statement, allowed_tables)

    # 6. Always bounded.
    statement, notes = _apply_row_limit(statement, max_rows)

    return GuardResult(
        safe_sql=statement.sql(dialect=DIALECT, pretty=True),
        original_sql=original,
        notes=notes,
        tables_used=tables_used,
    )


def is_safe(sql: str, allowed_tables: list[str] | None = None) -> bool:
    """Convenience wrapper for tests: True/False instead of an exception."""
    try:
        check_sql(sql, allowed_tables=allowed_tables)
        return True
    except UnsafeSQLError:
        return False
