"""
The tools the agent is allowed to use.

This is the part that makes DataPilot an *agent* rather than a chatbot. A
chatbot answers from whatever is in its context. An agent is given a list of
tools, decides for itself which to call and with what arguments, sees the
result, and decides what to do next.

The LLM never touches the data. It can only ask for one of these four things:

    list_tables()                      -> which tables exist
    describe_table(table_name)         -> columns, types, example values
    run_sql(query)                     -> run a read-only query, see the rows
    make_chart(chart_type, x, y, title)-> ask for a chart of the last result

Every tool returns **text**, because text is all the model can read. run_sql
returns either a small markdown table or an error message -- and that error
message is the whole trick behind self-correction: a failed query comes back
as a readable explanation, so on its next turn the model can fix it.

ToolBox also remembers the last successful query result. That matters because
make_chart cannot hand a picture to the model; it records what chart was asked
for, and the UI draws it afterwards from the stored result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.data_loader import DataStore
from src.executor import Executor, QueryError, QueryResult
from src.schema import describe_table
from src.sql_guard import UnsafeSQLError

# How many result rows to show the model. Enough to reason about, small enough
# to leave room in the context window.
ROWS_SHOWN_TO_MODEL = 20


# ---------------------------------------------------------------------------
# The tool definitions, in the JSON-schema format the Groq API expects
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "List the tables available, with their row and column counts. "
                "Use this first if you are unsure what data exists."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Show one table in detail: every column with its type, the range "
                "of numbers and dates, the possible values of short text columns, "
                "and three example rows. Use this before writing SQL if you are "
                "not certain a column exists or what its values look like."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The exact name of the table to describe.",
                    },
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run one read-only SQL SELECT query (DuckDB syntax) and get the "
                "rows back. This is the only way to obtain real numbers. If the "
                "query fails you will be told exactly why, and you should fix the "
                "query and call this tool again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A single SELECT statement (it may start with WITH). "
                            "No semicolons separating several statements, and no "
                            "statement that changes data."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Ask for a chart of the most recent query result. Call this only "
                "when a chart genuinely helps -- comparing categories, or showing "
                "a change over time. Do not chart a single number. The chart is "
                "shown to the user; you do not see it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "barh", "line", "area", "scatter"],
                        "description": (
                            "bar for comparing categories, barh when there are many "
                            "categories or long labels, line for change over time, "
                            "area for one series over time, scatter for two numbers."
                        ),
                    },
                    "x": {
                        "type": "string",
                        "description": "Column from the last result to use for the x-axis.",
                    },
                    "y": {
                        "type": "string",
                        "description": "Column from the last result to use for the y-axis.",
                    },
                    "title": {
                        "type": "string",
                        "description": "A short title describing what the chart shows.",
                    },
                },
                "required": ["chart_type"],
            },
        },
    },
]

TOOL_NAMES = [definition["function"]["name"] for definition in TOOL_DEFINITIONS]


# ---------------------------------------------------------------------------
# A record of each call, for the "How I got this" panel and the logs
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """One tool call: what was asked, what came back, and whether it worked."""
    name: str
    arguments: dict[str, Any]
    ok: bool
    observation: str                  # the text handed back to the model
    error: str | None = None
    duration_seconds: float = 0.0

    def short_description(self) -> str:
        """A one-line summary for the UI."""
        if self.name == "run_sql":
            query = str(self.arguments.get("query", "")).strip().replace("\n", " ")
            return f"run_sql: {query[:90]}"
        if self.name == "describe_table":
            return f"describe_table: {self.arguments.get('table_name', '?')}"
        if self.name == "make_chart":
            return f"make_chart: {self.arguments.get('chart_type', '?')}"
        return self.name


@dataclass
class ChartRequest:
    """What the model asked make_chart for."""
    chart_type: str
    x: str | None = None
    y: str | None = None
    title: str = ""


# ---------------------------------------------------------------------------
# The toolbox
# ---------------------------------------------------------------------------

@dataclass
class ToolBox:
    """
    Holds the data and runs the tools.

    One ToolBox serves one question. It keeps the last successful result and
    the last chart request, which is how the UI knows what table and chart to
    show when the agent has finished talking.
    """
    store: DataStore
    executor: Executor
    calls: list[ToolCallRecord] = field(default_factory=list)
    results: list[QueryResult] = field(default_factory=list)   # every successful query
    last_result: QueryResult | None = None
    chart_request: ChartRequest | None = None
    chart_result: QueryResult | None = None   # the result the chart is drawn from
    failed_sql_attempts: int = 0        # how many times the agent had to fix itself

    @classmethod
    def for_store(cls, store: DataStore, **executor_options: Any) -> "ToolBox":
        return cls(store=store, executor=Executor(store, **executor_options))

    def definitions(self) -> list[dict[str, Any]]:
        return TOOL_DEFINITIONS

    def close(self) -> None:
        self.executor.close()

    # -- the tools themselves ------------------------------------------------

    def _list_tables(self) -> str:
        if not self.store.tables:
            return "There are no tables loaded."
        lines = ["These tables are available:"]
        for table in self.store.tables.values():
            columns = ", ".join(str(c) for c in table.frame.columns)
            lines.append(
                f'- "{table.name}": {table.n_rows:,} rows, {table.n_cols} columns '
                f"({columns})"
            )
        return "\n".join(lines)

    def _describe_table(self, table_name: str) -> str:
        table = self.store.get(table_name)
        if table is None:
            available = ", ".join(f'"{n}"' for n in self.store.table_names())
            raise ValueError(
                f'There is no table called "{table_name}". '
                f"The tables available are: {available}."
            )
        return describe_table(table, include_value_hints=True)

    def _run_sql(self, query: str) -> str:
        """
        Run a query. On failure, raise with a readable message.

        The agent loop turns that message into the tool's reply, which is
        exactly what lets the model repair its own SQL.
        """
        try:
            result = self.executor.run(query)
        except (UnsafeSQLError, QueryError) as error:
            self.failed_sql_attempts += 1
            raise ValueError(str(error)) from error

        self.last_result = result
        self.results.append(result)

        header = f"The query returned {result.row_count:,} row(s)."
        if result.notes:
            header += " " + " ".join(result.notes)
        if result.is_empty:
            header += (
                " No rows matched. That usually means a filter value does not exist in "
                "the data -- check the possible values in the schema before retrying."
            )
        return f"{header}\n\n{result.to_markdown(max_rows=ROWS_SHOWN_TO_MODEL)}"

    def _make_chart(self, chart_type: str, x: str | None, y: str | None, title: str) -> str:
        """
        Prepare a chart of a query result.

        Subtle but important: the agent often runs one more query *after* the
        one it actually wants to chart (say, a follow-up total). Charting
        simply "the last result" therefore drew the wrong table -- and
        labelled it with axes that were not even in it.

        So instead of assuming, we search backwards through the results for
        the most recent one that genuinely contains the requested columns. If
        no result has them, that is reported as an error, which sends the
        agent round the loop again to re-run the query it meant to chart.
        """
        usable = [result for result in self.results if not result.is_empty]
        if not usable:
            raise ValueError(
                "There is no query result to chart yet. Call run_sql first, and "
                "make sure it returns rows."
            )

        wanted = [str(column) for column in (x, y) if column]
        chosen: QueryResult | None = None
        for result in reversed(usable):
            if all(column in result.column_names for column in wanted):
                chosen = result
                break

        if chosen is None:
            latest_columns = ", ".join(usable[-1].column_names)
            raise ValueError(
                f"No query result has the columns you asked to chart ({', '.join(wanted)}). "
                f"The most recent result has these columns: {latest_columns}. "
                "Either chart those columns, or run the query you want to chart again."
            )

        self.chart_request = ChartRequest(
            chart_type=str(chart_type or "").strip(),
            x=(x or None), y=(y or None), title=(title or ""),
        )
        self.chart_result = chosen

        note = ""
        if chosen is not usable[-1]:
            note = (" It uses the earlier result that actually contains those "
                    "columns, not your most recent query.")
        return (
            f"A {self.chart_request.chart_type} chart has been prepared and will be "
            f"shown to the user.{note} Now write your final answer in words -- do not "
            "call make_chart again."
        )

    # -- dispatch ------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallRecord:
        """
        Run one tool by name and always return a record -- never raise.

        Returning a record for failures too is the point: the agent must be
        able to *see* that something went wrong in order to fix it.
        """
        arguments = arguments or {}
        started = time.perf_counter()

        try:
            if name == "list_tables":
                observation = self._list_tables()
            elif name == "describe_table":
                observation = self._describe_table(str(arguments.get("table_name", "")))
            elif name == "run_sql":
                observation = self._run_sql(str(arguments.get("query", "")))
            elif name == "make_chart":
                observation = self._make_chart(
                    chart_type=str(arguments.get("chart_type", "")),
                    x=arguments.get("x"),
                    y=arguments.get("y"),
                    title=str(arguments.get("title", "")),
                )
            else:
                raise ValueError(
                    f"There is no tool called '{name}'. "
                    f"The tools available are: {', '.join(TOOL_NAMES)}."
                )

            record = ToolCallRecord(
                name=name, arguments=arguments, ok=True, observation=observation,
                duration_seconds=time.perf_counter() - started,
            )

        except Exception as error:   # noqa: BLE001 - reported back to the model
            message = str(error) or type(error).__name__
            record = ToolCallRecord(
                name=name, arguments=arguments, ok=False,
                observation=f"ERROR: {message}", error=message,
                duration_seconds=time.perf_counter() - started,
            )

        self.calls.append(record)
        return record
