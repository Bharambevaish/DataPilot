"""
Turn a query result into a sensible chart.

The agent can ask for a specific chart, but it does not have to: if it just
says "chart this", the shape of the result decides the chart type. That keeps
the agent's job simple and stops it asking for charts that make no sense.

The rules below are not personal taste, they follow standard data-visualisation
practice:

  * One number is not a chart. A single value is shown as a number.
  * Something over time is a line chart.
  * Comparing sizes across categories is a bar chart, in one colour. The bar
    length already shows the size, so giving each bar its own colour adds
    nothing and makes the chart harder to read.
  * Many categories go horizontal, so the labels stay readable.
  * Two numbers against each other is a scatter plot.
  * Distinct series (e.g. revenue per region per month) get the categorical
    palette, in a fixed order, capped at six -- the rest fold into "Other".
  * Pie and donut charts are never used. A request for one becomes ranked
    horizontal bars, which let you compare and rank the parts accurately.
  * Two measures on different scales never share one chart (no second y-axis),
    because that lets you imply any relationship you like by rescaling.

Colours come from a palette checked with a colour-blindness simulator, so
adjacent colours stay distinguishable. Three of the light-mode colours sit
below the 3:1 contrast guideline; that is acceptable here because DataPilot
always shows the result table next to the chart, so colour is never the only
way to read a value.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Colours. Light and dark are two chosen sets, not one set brightened.
# ---------------------------------------------------------------------------

CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#6250d6", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

THEME = {
    "light": {
        "series": CATEGORICAL_LIGHT,
        "single": "#2a78d6",      # one-hue blue for magnitude comparisons
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#898781",       # axis labels and ticks stay recessive
        "grid": "#e1e0d9",        # hairline gridlines
        "axis": "#c3c2b7",
    },
    "dark": {
        "series": CATEGORICAL_DARK,
        "single": "#3987e5",
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

# Above this many distinct categories, bars go horizontal so labels fit.
HORIZONTAL_ABOVE = 8

# Hard cap on how many bars/points to draw, so a 1,000-row result stays readable.
MAX_CATEGORIES = 25

# At most this many separate series; anything past it folds into "Other".
MAX_SERIES = 6


@dataclass
class ChartSpec:
    """The decision about what to draw, kept separate from the drawing."""
    chart_type: str                 # "bar" | "barh" | "line" | "area" | "scatter" | "none"
    x: str | None = None
    y: str | None = None
    series: str | None = None       # column that splits the data into series
    title: str = ""
    reason: str = ""                # why this chart, shown in the UI's "How I got this"


# ---------------------------------------------------------------------------
# Working out what the columns are
# ---------------------------------------------------------------------------

def _is_time(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _is_number(series: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series))


def _classify(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split the columns into time columns, number columns and label columns."""
    time_columns, number_columns, label_columns = [], [], []
    for column in frame.columns:
        series = frame[column]
        if _is_time(series):
            time_columns.append(str(column))
        elif _is_number(series):
            number_columns.append(str(column))
        else:
            label_columns.append(str(column))
    return time_columns, number_columns, label_columns


def _looks_like_a_year(frame: pd.DataFrame, column: str) -> bool:
    """
    Spot a number column that is really a time axis, e.g. year or month number.

    GROUP BY YEAR(date) gives a number, but it should still be a line chart.
    """
    name = column.lower()
    if name in {"year", "yr", "month", "month_number", "quarter", "week",
                "day", "hour", "yearmonth", "year_month"}:
        return True
    series = frame[column].dropna()
    if series.empty or not _is_number(series):
        return False
    # Whole numbers in a plausible year range, all different.
    looks_whole = (series % 1 == 0).all()
    return bool(looks_whole and series.between(1900, 2100).all()
                and series.is_unique and len(series) > 2)


# ---------------------------------------------------------------------------
# Choosing the chart
# ---------------------------------------------------------------------------

def suggest_chart(frame: pd.DataFrame) -> ChartSpec:
    """Decide what chart (if any) suits this result, with a reason."""
    if frame is None or frame.empty:
        return ChartSpec("none", reason="There are no rows to chart.")

    n_rows, n_cols = frame.shape
    time_columns, number_columns, label_columns = _classify(frame)

    # One value: the answer is a number, not a picture.
    if n_rows == 1 and n_cols == 1:
        return ChartSpec("none", reason="The answer is a single number, so no chart is needed.")

    if not number_columns:
        return ChartSpec("none", reason="There is nothing numeric to measure, so no chart is drawn.")

    # A number column that is really a year/month behaves like a time axis.
    year_like = [c for c in number_columns if _looks_like_a_year(frame, c)]
    measures = [c for c in number_columns if c not in year_like]
    if not measures:
        return ChartSpec("none", reason="The only numbers look like dates, so there is nothing to measure.")

    measure = measures[0]   # one measure only -- never two scales on one chart

    # Time on the x-axis -> line chart.
    x_time = time_columns[0] if time_columns else (year_like[0] if year_like else None)
    if x_time:
        # A third column that splits the line into several series.
        series_column = label_columns[0] if label_columns else None
        if series_column and frame[series_column].nunique() > MAX_SERIES * 3:
            series_column = None  # far too many lines to read
        return ChartSpec(
            "line", x=x_time, y=measure, series=series_column,
            reason=f"'{x_time}' is a time axis, so changes over time are drawn as a line.",
        )

    # Categories on the x-axis -> bar chart.
    if label_columns:
        category = label_columns[0]
        series_column = label_columns[1] if len(label_columns) > 1 else None
        n_categories = frame[category].nunique()
        if n_categories > HORIZONTAL_ABOVE:
            return ChartSpec(
                "barh", x=measure, y=category, series=series_column,
                reason=(f"There are {n_categories} categories, so the bars run "
                        "horizontally to keep the labels readable."),
            )
        return ChartSpec(
            "bar", x=category, y=measure, series=series_column,
            reason=f"Comparing '{measure}' across {n_categories} categories, so a bar chart.",
        )

    # Two numbers and nothing else -> scatter.
    if len(measures) >= 2:
        return ChartSpec(
            "scatter", x=measures[0], y=measures[1],
            reason="Two numeric columns, so they are plotted against each other.",
        )

    return ChartSpec("none", reason="The shape of this result does not suit a chart.")


# ---------------------------------------------------------------------------
# Drawing it
# ---------------------------------------------------------------------------

def _style(figure: go.Figure, spec: ChartSpec, dark: bool, show_legend: bool) -> go.Figure:
    """Apply the shared look: recessive axes, hairline grid, readable hover."""
    colours = THEME["dark" if dark else "light"]

    figure.update_layout(
        title=dict(text=spec.title or "", font=dict(size=15, color=colours["text"])),
        paper_bgcolor=colours["surface"],
        plot_bgcolor=colours["surface"],
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            size=12, color=colours["text"],
        ),
        margin=dict(l=10, r=10, t=44 if spec.title else 16, b=10),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0, title_text="",
                    font=dict(size=11, color=colours["text"])),
        hoverlabel=dict(
            bgcolor=colours["surface"], bordercolor=colours["axis"],
            font=dict(color=colours["text"], size=12),
        ),
        # One shared crosshair tooltip on lines; per-mark tooltips elsewhere.
        hovermode="x unified" if spec.chart_type in {"line", "area"} else "closest",
    )

    axis_style = dict(
        showgrid=True, gridcolor=colours["grid"], gridwidth=1,
        zeroline=False, linecolor=colours["axis"],
        tickfont=dict(color=colours["muted"], size=11),
        title_font=dict(color=colours["muted"], size=11),
    )
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)

    # Category axes do not need gridlines in the category direction.
    if spec.chart_type == "bar":
        figure.update_xaxes(showgrid=False)
    elif spec.chart_type == "barh":
        figure.update_yaxes(showgrid=False)

    return figure


def _trim(frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep the chart readable: cap the categories, and fold a long tail of
    series into "Other" rather than drawing 40 colours.
    """
    notes: list[str] = []
    frame = frame.copy()

    # Too many series -> keep the biggest few, fold the rest together.
    if spec.series and spec.series in frame.columns:
        measure = spec.y if spec.chart_type != "barh" else spec.x
        if measure and measure in frame.columns:
            totals = frame.groupby(spec.series)[measure].sum().sort_values(ascending=False)
            if len(totals) > MAX_SERIES:
                keep = set(totals.head(MAX_SERIES).index)
                frame[spec.series] = frame[spec.series].where(
                    frame[spec.series].isin(keep), "Other"
                )
                notes.append(
                    f"Only the {MAX_SERIES} largest '{spec.series}' values are drawn "
                    'separately; the rest are grouped as "Other".'
                )

    # Too many bars/points -> keep the largest ones.
    category = spec.x if spec.chart_type in {"bar", "line"} else spec.y
    if category and category in frame.columns and spec.chart_type in {"bar", "barh"}:
        measure = spec.y if spec.chart_type == "bar" else spec.x
        if frame[category].nunique() > MAX_CATEGORIES and measure in frame.columns:
            top = (frame.groupby(category)[measure].sum()
                   .sort_values(ascending=False).head(MAX_CATEGORIES).index)
            frame = frame[frame[category].isin(top)]
            notes.append(f"Showing the {MAX_CATEGORIES} largest '{category}' values only.")

    return frame, notes


def _series_groups(frame: pd.DataFrame, spec: ChartSpec) -> list[tuple[str | None, pd.DataFrame]]:
    """Split the data into (series name, rows) pairs, in a stable order."""
    if not spec.series or spec.series not in frame.columns:
        return [(None, frame)]

    measure = spec.y if spec.chart_type != "barh" else spec.x
    order = (frame.groupby(spec.series)[measure].sum().sort_values(ascending=False).index
             if measure in frame.columns else sorted(frame[spec.series].unique()))
    # "Other" always goes last, whatever its size.
    names = [n for n in order if n != "Other"] + (["Other"] if "Other" in set(order) else [])
    return [(str(name), frame[frame[spec.series] == name]) for name in names]


def build_chart(
    frame: pd.DataFrame,
    chart_type: str | None = None,
    x: str | None = None,
    y: str | None = None,
    title: str = "",
    dark: bool = False,
) -> tuple[go.Figure | None, ChartSpec]:
    """
    Build a Plotly figure for this result.

    Any of chart_type/x/y may come from the agent. Anything it leaves out, or
    gets wrong, falls back to what the data shape suggests -- so a confused
    agent still produces a correct chart instead of an error.

    Returns (figure or None, the spec that was used). A None figure is normal,
    not a failure: it means this result is better read as a number or a table.
    """
    spec = suggest_chart(frame)

    requested = (chart_type or "").strip().lower().replace("_", "").replace(" ", "")
    aliases = {
        "bar": "bar", "column": "bar", "columns": "bar", "verticalbar": "bar",
        "barh": "barh", "horizontalbar": "barh", "hbar": "barh", "barhorizontal": "barh",
        "line": "line", "lines": "line", "timeseries": "line", "trend": "line",
        "area": "area", "stackedarea": "area",
        "scatter": "scatter", "point": "scatter", "points": "scatter", "bubble": "scatter",
        "none": "none", "table": "none", "no": "none",
    }

    # A pie or donut request becomes ranked horizontal bars.
    if requested in {"pie", "donut", "doughnut", "piechart"}:
        if spec.chart_type in {"bar", "barh"}:
            category = spec.x if spec.chart_type == "bar" else spec.y
            measure = spec.y if spec.chart_type == "bar" else spec.x
            spec = ChartSpec(
                "barh", x=measure, y=category, series=spec.series, title=title,
                reason=("Shown as ranked bars rather than a pie chart: bars on a "
                        "shared baseline let you compare and rank the parts accurately."),
            )
    elif requested in aliases:
        wanted = aliases[requested]
        if wanted == "none":
            spec = ChartSpec("none", reason="No chart was asked for.")
        elif spec.chart_type != "none":
            # Honour the request, but keep the columns the data supports.
            spec = ChartSpec(
                wanted, x=spec.x, y=spec.y, series=spec.series,
                reason=f"Drawn as a {wanted} chart, as requested.",
            )

    # Let the agent name the axes, but only with columns that really exist.
    if x and x in frame.columns and spec.chart_type != "none":
        spec.x = x
    if y and y in frame.columns and spec.chart_type != "none":
        spec.y = y
    spec.title = title or spec.title

    if spec.chart_type == "none":
        return None, spec

    # Axis columns must exist; if the fallback left them broken, draw nothing.
    needed = [c for c in (spec.x, spec.y) if c is not None]
    if len(needed) < 2 or any(c not in frame.columns for c in needed):
        return None, ChartSpec("none", reason="The columns needed for a chart are not in the result.")

    plot_frame, trim_notes = _trim(frame, spec)
    if trim_notes:
        spec.reason = " ".join([spec.reason, *trim_notes])

    colours = THEME["dark" if dark else "light"]
    groups = _series_groups(plot_frame, spec)
    multi = len(groups) > 1

    figure = go.Figure()
    for index, (name, part) in enumerate(groups):
        # Fixed colour order, never cycled: past the palette we stop adding series.
        colour = colours["series"][index] if multi else colours["single"]
        label = name or str(spec.y)

        if spec.chart_type == "bar":
            figure.add_bar(
                x=part[spec.x], y=part[spec.y], name=label,
                marker=dict(color=colour, line=dict(width=0)),
                hovertemplate=f"%{{x}}<br>{spec.y}: %{{y:,.2f}}<extra>{label}</extra>",
            )
        elif spec.chart_type == "barh":
            ordered = part.sort_values(spec.x, ascending=True)  # largest at the top
            figure.add_bar(
                x=ordered[spec.x], y=ordered[spec.y], orientation="h", name=label,
                marker=dict(color=colour, line=dict(width=0)),
                hovertemplate=f"%{{y}}<br>{spec.x}: %{{x:,.2f}}<extra>{label}</extra>",
            )
        elif spec.chart_type in {"line", "area"}:
            ordered = part.sort_values(spec.x)
            figure.add_scatter(
                x=ordered[spec.x], y=ordered[spec.y], name=label, mode="lines",
                line=dict(color=colour, width=2),
                fill="tozeroy" if spec.chart_type == "area" and not multi else None,
                hovertemplate=f"{spec.y}: %{{y:,.2f}}<extra>{label}</extra>",
            )
        else:  # scatter
            figure.add_scatter(
                x=part[spec.x], y=part[spec.y], name=label, mode="markers",
                marker=dict(color=colour, size=9,
                            line=dict(width=1, color=colours["surface"])),
                hovertemplate=(f"{spec.x}: %{{x:,.2f}}<br>{spec.y}: %{{y:,.2f}}"
                               f"<extra>{label}</extra>"),
            )

    if spec.chart_type == "bar" and multi:
        figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.04)
    elif spec.chart_type == "barh" and multi:
        figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.04)

    # A legend is only meaningful with more than one series; the title names a single one.
    figure = _style(figure, spec, dark=dark, show_legend=multi)

    # Axis titles, written the way the column is named in the result.
    figure.update_xaxes(title_text=str(spec.x))
    figure.update_yaxes(title_text=str(spec.y))

    return figure, spec
