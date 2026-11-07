"""
Turn uploaded files into SQL tables.

The job of this file is small but important: take a CSV or Excel file that a
human made, and produce clean DuckDB tables the agent can query reliably.

Real spreadsheets are messy. Column headers look like "Total Sales (INR)" or
"Order Date". SQL hates those, and so does an LLM writing SQL. So every column
name is rewritten to a plain snake_case name before anything else happens.

We write the tables into a DuckDB *file* (not an in-memory database) on
purpose: a file can later be re-opened as read-only, which is one of the three
safety layers described in src/sql_guard.py and src/executor.py.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

# Extensions we know how to read.
CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}

# Words that hint a text column really holds dates.
_DATE_NAME_HINTS = ("date", "day", "month", "year", "time", "timestamp",
                    "created", "updated", "joined", "signup")

# A value that looks like 2024-01-31, 31/01/2024, 01-31-2024, 2024/01/31 ...
_DATE_LIKE = re.compile(
    r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$"
)


class DataLoadError(Exception):
    """Raised when a file cannot be turned into tables, with a readable reason."""


# ---------------------------------------------------------------------------
# Name cleaning
# ---------------------------------------------------------------------------

def clean_name(raw: str) -> str:
    """
    Turn any header or sheet name into a safe snake_case SQL identifier.

        "Total Sales (INR)"  -> "total_sales_inr"
        "Order Date"         -> "order_date"
        "2024 Q1"            -> "col_2024_q1"
        ""                   -> "unnamed"
    """
    name = str(raw).strip().lower()
    # Anything that is not a letter, digit or underscore becomes an underscore.
    name = re.sub(r"[^0-9a-z_]+", "_", name)
    # Collapse runs of underscores and trim them from the ends.
    name = re.sub(r"_+", "_", name).strip("_")

    if not name:
        return "unnamed"
    # SQL identifiers may not start with a digit.
    if name[0].isdigit():
        name = f"col_{name}"
    return name


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename every column with clean_name, keeping names unique."""
    seen: dict[str, int] = {}
    new_names: list[str] = []

    for column in frame.columns:
        name = clean_name(column)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"  # total, total_2, total_3 ...
        else:
            seen[name] = 1
        new_names.append(name)

    frame = frame.copy()
    frame.columns = new_names
    return frame


# ---------------------------------------------------------------------------
# Date detection
# ---------------------------------------------------------------------------

def _looks_like_dates(series: pd.Series, column_name: str) -> bool:
    """
    Decide whether a text column should become a real date column.

    Two things must be true, which keeps us from wrecking id columns:
      1. the values actually look like dates (or the column is named like one),
      2. at least 90% of the non-empty values parse as dates.
    """
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False

    name_hint = any(hint in column_name for hint in _DATE_NAME_HINTS)
    shape_hint = sample.str.match(_DATE_LIKE).mean() >= 0.9

    # An id like "S100001" has neither hint, so it is left alone.
    if not (name_hint or shape_hint):
        return False

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def _convert_dates(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert text date columns to real dates. Returns the frame and the names changed."""
    frame = frame.copy()
    converted: list[str] = []

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            converted.append(column)
            continue
        # Only consider text columns -- never touch numbers.
        if series.dtype != object and not isinstance(series.dtype, pd.StringDtype):
            continue
        if _looks_like_dates(series, column):
            frame[column] = pd.to_datetime(series, errors="coerce", format="mixed")
            converted.append(column)

    return frame, converted


# ---------------------------------------------------------------------------
# What a loaded dataset looks like
# ---------------------------------------------------------------------------

@dataclass
class LoadedTable:
    """One SQL table, plus where it came from."""
    name: str
    frame: pd.DataFrame
    source: str                       # "sales.csv" or "shop_orders.xlsx (sheet: orders)"
    date_columns: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_cols(self) -> int:
        return len(self.frame.columns)


@dataclass
class DataStore:
    """
    Everything the agent is allowed to see: a DuckDB file plus its tables.

    `db_path` points at a DuckDB database on disk. src/executor.py re-opens
    that same path in read-only mode, so the agent physically cannot write to
    it even if a dangerous query slipped past the SQL checks.
    """
    db_path: Path
    tables: dict[str, LoadedTable] = field(default_factory=dict)
    _temp_dir: Path | None = None

    def table_names(self) -> list[str]:
        return list(self.tables)

    def get(self, name: str) -> LoadedTable | None:
        """Look up a table, ignoring case so 'Sales' finds 'sales'."""
        if name in self.tables:
            return self.tables[name]
        lowered = str(name).strip().strip('"').lower()
        for key, table in self.tables.items():
            if key.lower() == lowered:
                return table
        return None

    def total_rows(self) -> int:
        return sum(table.n_rows for table in self.tables.values())

    def cleanup(self) -> None:
        """Delete the temporary DuckDB file. Safe to call more than once."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV, trying a couple of encodings before giving up."""
    separator = "\t" if path.suffix.lower() == ".tsv" else None  # None = sniff it
    last_error: Exception | None = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=separator, encoding=encoding, engine="python")
        except UnicodeDecodeError as error:
            last_error = error
        except Exception as error:
            raise DataLoadError(f"Could not read {path.name}: {error}") from error

    raise DataLoadError(f"Could not read {path.name}: unknown text encoding ({last_error})")


def _frames_from_file(path: Path) -> dict[str, tuple[pd.DataFrame, str]]:
    """
    Read one file into {table_name: (dataframe, human readable source)}.

    A CSV gives one table. An Excel workbook gives one table per sheet.
    """
    suffix = path.suffix.lower()
    stem = clean_name(path.stem)

    if suffix in CSV_SUFFIXES:
        return {stem: (_read_csv(path), path.name)}

    if suffix in EXCEL_SUFFIXES:
        try:
            sheets = pd.read_excel(path, sheet_name=None)  # None = every sheet
        except Exception as error:
            raise DataLoadError(f"Could not read {path.name}: {error}") from error

        out: dict[str, tuple[pd.DataFrame, str]] = {}
        for sheet_name, frame in sheets.items():
            if frame.empty:
                continue  # skip blank sheets rather than creating useless tables
            # One sheet -> use the file name; several -> use the sheet names.
            table = clean_name(sheet_name) if len(sheets) > 1 else stem
            out[table] = (frame, f"{path.name} (sheet: {sheet_name})")
        if not out:
            raise DataLoadError(f"{path.name} has no sheets with any data in them.")
        return out

    raise DataLoadError(
        f"{path.name}: unsupported file type '{suffix}'. "
        "Please upload a .csv, .tsv or .xlsx file."
    )


def _close_quietly(connection: duckdb.DuckDBPyConnection) -> None:
    """Close a DuckDB connection, ignoring an already-closed connection."""
    try:
        connection.close()
    except Exception:
        pass


def load_files(paths: list[str | Path]) -> DataStore:
    """
    Load one or more files into a fresh read-write DuckDB file.

    This is the only place that writes to the database. Everything afterwards
    reads from it through a read-only connection.
    """
    if not paths:
        raise DataLoadError("No files were given to load.")

    temp_dir = Path(tempfile.mkdtemp(prefix="datapilot_"))
    db_path = temp_dir / "datapilot.duckdb"
    store = DataStore(db_path=db_path, _temp_dir=temp_dir)

    connection = duckdb.connect(str(db_path))
    try:
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                raise DataLoadError(f"File not found: {path}")

            for table_name, (frame, source) in _frames_from_file(path).items():
                if frame.empty:
                    continue

                frame = _clean_columns(frame)
                # Drop columns pandas invents for empty spreadsheet headers.
                frame = frame.loc[:, ~frame.columns.str.startswith("unnamed")]
                frame, date_columns = _convert_dates(frame)

                # Avoid clashes when two files produce the same table name.
                final_name = table_name
                suffix_number = 2
                while final_name in store.tables:
                    final_name = f"{table_name}_{suffix_number}"
                    suffix_number += 1

                # Hand the DataFrame to DuckDB and copy it into a real table.
                connection.register("incoming_frame", frame)
                connection.execute(
                    f'CREATE OR REPLACE TABLE "{final_name}" AS '
                    "SELECT * FROM incoming_frame"
                )
                connection.unregister("incoming_frame")

                store.tables[final_name] = LoadedTable(
                    name=final_name,
                    frame=frame,
                    source=source,
                    date_columns=date_columns,
                )
    except Exception:
        _close_quietly(connection)
        store.cleanup()
        raise

    # Close the writing connection. The database file must not be held open
    # here, because executor.py re-opens this same path in read-only mode.
    _close_quietly(connection)

    if not store.tables:
        store.cleanup()
        raise DataLoadError("The file(s) loaded, but contained no usable tables.")

    return store


def load_sample_dataset(name: str) -> DataStore:
    """
    Load one of the bundled sample datasets by short name.

    "sales"       -> data/samples/sales.csv
    "shop_orders" -> data/samples/shop_orders.xlsx (both sheets)
    """
    from src import config

    options = {
        "sales": [config.SAMPLES_DIR / "sales.csv"],
        "shop_orders": [config.SAMPLES_DIR / "shop_orders.xlsx"],
    }
    if name not in options:
        raise DataLoadError(
            f"Unknown sample dataset '{name}'. Choose one of: {', '.join(options)}"
        )

    missing = [p for p in options[name] if not p.exists()]
    if missing:
        raise DataLoadError(
            "Sample data is missing. Run this first:  python scripts/make_samples.py"
        )

    return load_files(options[name])
