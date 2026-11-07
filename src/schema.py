"""
Describe the loaded tables in plain text, for the LLM to read.

This file is the single most important one for accuracy. The agent never sees
the data -- it only sees the text produced here. If this description is vague,
the agent guesses column names and writes broken SQL. If it is precise, the
agent usually gets the query right first time.

Three things go into the description:

1. Structure   -- table names, column names, column types.
2. Value hints -- for a column with only a few distinct values we list them,
                  so the agent writes  region = 'South'  and not  region = 'south'.
                  For numbers and dates we give the min and max, so the agent
                  knows the data covers 2024 and not 2019.
3. Sample rows -- three real rows, so the agent can see the shape of the data.

`include_value_hints` can be turned off. That is not decoration: the
evaluation in eval/evaluate.py measures accuracy with it on and off, which is
the "before and after" experiment in eval/results.md.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import DataStore, LoadedTable

# A text column with at most this many distinct values gets its values listed.
MAX_DISTINCT_TO_LIST = 15

# How many example rows to show per table.
SAMPLE_ROWS = 3


def sql_type_name(series: pd.Series) -> str:
    """Translate a pandas dtype into the SQL type name DuckDB would use."""
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "VARCHAR"


def _format_number(value: float) -> str:
    """Format a number so the prompt stays short and readable."""
    if pd.isna(value):
        return "null"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _format_date(value) -> str:
    """Show a date as YYYY-MM-DD, dropping a midnight time part."""
    timestamp = pd.Timestamp(value)
    if timestamp.hour or timestamp.minute or timestamp.second:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp.strftime("%Y-%m-%d")


def describe_column(series: pd.Series, name: str, include_value_hints: bool = True) -> str:
    """
    Build one line describing one column, for example:

        region        VARCHAR    4 distinct: 'East', 'North', 'South', 'West'
        unit_price    DOUBLE     min 524.47, max 102,363.34
        date          TIMESTAMP  from 2024-01-01 to 2024-12-31
        order_id      VARCHAR    4,974 distinct values
    """
    type_name = sql_type_name(series)
    line = f"  - {name} ({type_name})"

    if not include_value_hints:
        return line

    notes: list[str] = []
    non_null = series.dropna()

    null_count = int(series.isna().sum())
    if null_count:
        notes.append(f"{null_count:,} empty values")

    if non_null.empty:
        notes.append("column is entirely empty")

    elif pd.api.types.is_datetime64_any_dtype(series):
        notes.append(f"from {_format_date(non_null.min())} to {_format_date(non_null.max())}")

    elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        notes.append(f"min {_format_number(non_null.min())}")
        notes.append(f"max {_format_number(non_null.max())}")
        notes.append(f"average {_format_number(non_null.mean())}")

    else:
        # Text (or boolean). If there are only a few values, list them all --
        # this is what stops the agent inventing categories that do not exist.
        distinct = non_null.unique()
        if len(distinct) <= MAX_DISTINCT_TO_LIST:
            shown = ", ".join(f"'{value}'" for value in sorted(map(str, distinct)))
            notes.append(f"{len(distinct)} possible values: {shown}")
        else:
            examples = ", ".join(f"'{value}'" for value in map(str, distinct[:3]))
            notes.append(f"{len(distinct):,} distinct values, e.g. {examples}")

    if notes:
        line += "  -- " + "; ".join(notes)
    return line


def _sample_rows_text(frame: pd.DataFrame, n_rows: int = SAMPLE_ROWS) -> str:
    """Render the first few rows as a compact pipe-separated table."""
    sample = frame.head(n_rows).copy()

    # Dates print as "2024-01-01", not "2024-01-01 00:00:00".
    for column in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[column]):
            sample[column] = sample[column].map(
                lambda v: "" if pd.isna(v) else _format_date(v)
            )

    header = " | ".join(str(c) for c in sample.columns)
    lines = [f"    {header}"]
    for _, row in sample.iterrows():
        lines.append("    " + " | ".join("" if pd.isna(v) else str(v) for v in row))
    return "\n".join(lines)


def describe_table(table: LoadedTable, include_value_hints: bool = True) -> str:
    """Build the full description of one table."""
    parts = [
        f'Table "{table.name}" -- {table.n_rows:,} rows, {table.n_cols} columns '
        f"(loaded from {table.source})",
        "Columns:",
    ]
    for column in table.frame.columns:
        parts.append(describe_column(table.frame[column], column, include_value_hints))

    if include_value_hints and table.n_rows:
        parts.append(f"First {min(SAMPLE_ROWS, table.n_rows)} rows:")
        parts.append(_sample_rows_text(table.frame))

    return "\n".join(parts)


def _guess_join_keys(store: DataStore) -> list[str]:
    """
    Guess which columns link two tables, and say so in the prompt.

    The rule is deliberately simple: a column name that appears in more than
    one table is probably a join key. It is only a hint, and it is labelled as
    a hint, so the agent can ignore it when it is wrong.
    """
    column_owners: dict[str, list[str]] = {}
    for table in store.tables.values():
        for column in table.frame.columns:
            column_owners.setdefault(column, []).append(table.name)

    hints: list[str] = []
    for column, owners in sorted(column_owners.items()):
        if len(owners) > 1:
            joined = " and ".join(f'"{o}"' for o in owners)
            hints.append(f"  - {column} appears in {joined}, so it probably joins them")
    return hints


def build_schema_text(store: DataStore, include_value_hints: bool = True) -> str:
    """
    Build the whole schema description that goes into the agent's prompt.

    Keeping this as one plain string (rather than JSON) is deliberate: it is
    far easier to eyeball when a query goes wrong, and it reads naturally to
    the model.
    """
    if not store.tables:
        return "No tables are loaded."

    blocks = [
        f"The database has {len(store.tables)} table(s), "
        f"{store.total_rows():,} rows in total.",
        "",
    ]

    for table in store.tables.values():
        blocks.append(describe_table(table, include_value_hints))
        blocks.append("")

    if len(store.tables) > 1:
        hints = _guess_join_keys(store)
        if hints:
            blocks.append("Likely relationships between tables (a hint, not a rule):")
            blocks.extend(hints)
            blocks.append("")

    return "\n".join(blocks).strip()


def schema_summary(store: DataStore) -> str:
    """One short line per table, for the UI sidebar and log files."""
    return "; ".join(
        f"{t.name} ({t.n_rows:,} rows x {t.n_cols} cols)" for t in store.tables.values()
    )
