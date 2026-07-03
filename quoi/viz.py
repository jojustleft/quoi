from copy import deepcopy
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
import warnings

__all__ = [
    "bold_text",
    "build_subtitle",
    "plot_line",
    "plot_bar",
    "plot_area",
    "setup_chart_template",
    "add_titles",
]

# TODO:
# Change common chart parameters (title, axis titles, ...) by kwargs
# Replace trim_x and trim_y behaviour in chart functions by standalone method

layout_config = dict(
    text="#201B23",
    background="#FCF6F5",
    grid="#ebe1df",
    entries=["#5C825B", "#D5834F", "#4E76B2", "#904A58", "#C0E0DE"],
    serif_font="Georgia",
    default_font="Helvetica",
    subtitle_size=20,
    subtitle_color="#4a464d",
    warning_color="#D5834F",
)

layout = dict(
    layout=go.Layout(
        paper_bgcolor=layout_config["background"],
        plot_bgcolor=layout_config["background"],
        title=dict(
            font=dict(family=layout_config["serif_font"], size=22, color=layout_config["text"]),
            xanchor="left",
            xref="paper",
            x=0,
            yanchor="bottom",
            yref="container",
            y=0.92,
            automargin=True,
            pad_b=10,
        ),
        xaxis=dict(
            title=dict(
                font=dict(
                    family=layout_config["default_font"],
                    size=16,
                    color=layout_config["text"],
                )
            ),
            showline=True,
            ticks="outside",
            mirror=True,
            zeroline=False,
            gridcolor=layout_config["grid"],
            automargin=True,
        ),
        yaxis=dict(
            title=dict(
                font=dict(
                    family=layout_config["default_font"],
                    size=16,
                    color=layout_config["text"],
                )
            ),
            showline=True,
            ticks="outside",
            mirror=True,
            zeroline=False,
            gridcolor=layout_config["grid"],
            automargin=True,
        ),
        legend=dict(
            title=dict(
                font=dict(
                    family=layout_config["default_font"],
                    size=15,
                    color=layout_config["text"],
                )
            ),
            orientation="h",
            xanchor="right",
            x=1,
            yanchor="bottom",
            y=1.02,
            font=dict(family=layout_config["default_font"]),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        hoverlabel=dict(align="left"),
        colorway=layout_config["entries"],
        margin=dict(t=75, b=75, l=75, r=75, pad=5),
    )
)


def setup_chart_template(from_notebook=False):
    pio.templates["quoi"] = layout
    pio.templates.default = "quoi"

    if from_notebook:
        pio.renderers.default = "notebook"


def build_subtitle(s):
    # Not the most elegant solution, but it is the approach which provided the layout I preferred
    return f"<br><span style='font-family: {layout_config['default_font']}; font-size: {layout_config['subtitle_size']}px; color:{layout_config['subtitle_color']}'><sup>{s}</sup></span>"


def bold_text(s):
    if "<b>" in s:
        warnings.warn("<b> tag already found in provided string")

    return f"<b>{s}</b>"


def build_default_hovertemplate(fig: go.Figure) -> go.Figure:
    legend_title = fig.layout.legend_title.replace("<b>", "").replace("</b>", "") if fig.layout.legend_title else ""

    hovertemplate = "<br>".join(
        [
            "%{x}",
            ("<b>" + legend_title + ":</b> %{fullData.name}") if legend_title else "",
            "<b>%{yaxis.title.text}:</b> %{y}<extra></extra>",
        ]
    ).replace("<br><br>", "<br>")

    trace_is_horizontal_bar = [(isinstance(el, go.Bar) and el.orientation == "h") for el in fig.data]

    if all(trace_is_horizontal_bar):
        hovertemplate = (
            hovertemplate.replace("%{x}", "%{y}").replace("%{yaxis", "%{xaxis").replace("%{y}<extra>", "%{x}<extra>")
        )
    elif any(trace_is_horizontal_bar):
        raise ValueError("Multiple visualization formats found in chart, cannot build default hovertemplate")

    return hovertemplate


def plot_line(
    df: pl.DataFrame,
    x: str,
    y: str,
    breakdown: str = None,
    title: str = None,
    subtitle: str = None,
    x_title: str = None,
    y_title: str = None,
    legend_title: str = None,
    legend_replace: dict = None,
    legend_summary: bool = False,
    return_fig: bool = False,
    hovertemplate: str = None,
):
    if title is None and subtitle is not None:
        raise ValueError("title must be provided when subtitle is not empty.")
    if breakdown and (isinstance(y, list) and len(y) > 1):
        raise ValueError("If breakdown is provided, y must be a single column")

    fig = go.Figure()

    dict_summary = dict()
    if legend_summary:
        if isinstance(y, str) and df[y].min() < 0:
            warnings.warn("y contains negative values, this will impact legend summary.")
        if isinstance(y, str) and not breakdown:
            warnings.warn("There is only one provided trace, skipping legend summary.")

        if isinstance(y, list):
            df_summary = (
                df.unpivot(on=y, variable_name="trace")
                .group_by("trace")
                .agg(pl.col("value").sum())
                .with_columns((pl.col("value") / pl.col("value").sum() * 100).alias("share"))
            )
        else:
            df_summary = df.group_by(breakdown).agg(pl.col(y).sum().alias("value")).rename({"breakdown": "trace"})
        df_summary = df_summary.with_columns((pl.col("value") / pl.col("value").sum() * 100).alias("share"))
        df_summary = df_summary[["trace", "share"]].to_dict(as_series=False)
        dict_summary = {k: v for k, v in zip(df_summary["trace"], df_summary["share"])}

    if breakdown is not None:
        for entry, group in df.group_by(breakdown):
            fig.add_trace(go.Scatter(x=group[x], y=group[y], name=entry[0]))

    else:
        cols_to_plot = [y] if isinstance(y, str) else y
        for el in cols_to_plot:
            fig.add_trace(go.Scatter(x=df[x], y=df[el], name=el))

    add_titles(
        fig,
        title=title,
        subtitle=subtitle,
        x_title=x_title,
        y_title=y_title,
        legend_title=legend_title,
    )

    if hovertemplate == "default":
        chart_hovertemplate = build_default_hovertemplate(fig)
    else:
        chart_hovertemplate = hovertemplate if hovertemplate is not None else None
    fig.update_traces(hovertemplate=chart_hovertemplate)

    # Replace legend entry names or add summary share
    if legend_summary or legend_replace:
        for i in range(len(fig.data)):
            curr_name = fig.data[i].name
            replace_name = legend_replace.get(curr_name, curr_name)
            curr_value = dict_summary.get(curr_name)
            curr_value_str = f" ({curr_value:.2f}%)" if curr_value else ""
            fig.data[i].name = replace_name + curr_value_str

    if return_fig:
        return fig
    fig.show()


def plot_bar(
    df: pl.DataFrame,
    x: str,
    y: str,
    text: str = None,
    breakdown: str = None,
    breakdown_stack: bool = False,
    summary_sum: bool = False,
    title: str = None,
    subtitle: str = None,
    x_title: str = None,
    y_title: str = None,
    legend_title: str = None,
    legend_replace: dict = None,
    return_fig: bool = False,
    hovertemplate: str = None,
    trim_x: int = None,
    trim_y: int = None,
    orientation="v",
):
    if title is None and subtitle is not None:
        raise ValueError("title must be provided when subtitle is not empty.")

    fig = go.Figure()

    if breakdown is not None:
        if summary_sum:
            df = df.with_columns((pl.col(y) / pl.col(y).sum() * 100).over(pl.col(x)).alias("__sum_share"))
        for entry, group in df.group_by(breakdown, maintain_order=True):
            curr_name = entry[0] if not legend_replace else legend_replace.get(entry[0], entry[0])
            trace_text = group[text] if text else None
            customdata = group["__sum_share"] if summary_sum else None
            fig.add_trace(
                go.Bar(
                    x=group[x],
                    y=group[y],
                    name=curr_name,
                    orientation=orientation,
                    text=trace_text,
                    customdata=customdata,
                )
            )

    else:
        if summary_sum:
            df = df.with_columns((pl.col(y) / pl.col(y).sum() * 100).alias("__sum_share"))
        curr_name = y if not legend_replace else legend_replace.get(y, y)
        trace_text = df[text] if text else None
        customdata = df["__sum_share"] if summary_sum else None
        fig.add_trace(
            go.Bar(
                x=df[x],
                y=df[y],
                name=curr_name,
                orientation=orientation,
                text=trace_text,
                customdata=customdata,
            )
        )

    add_titles(
        fig,
        title=title,
        subtitle=subtitle,
        x_title=x_title,
        y_title=y_title,
        legend_title=legend_title,
    )

    chart_barmode = "stack" if breakdown_stack else None
    fig.update_layout(barmode=chart_barmode)

    if hovertemplate == "default":
        chart_hovertemplate = build_default_hovertemplate(fig)
    else:
        chart_hovertemplate = hovertemplate if hovertemplate is not None else None
    fig.update_traces(hovertemplate=chart_hovertemplate)

    # This behaviour should probably be on its own method (along with trim_legend)
    if trim_x:
        for i in range(len(fig.data)):
            original_values = fig.data[i].x
            fig.data[i].x = [el if len(el) <= trim_x else f"{el[:trim_x]}..." for el in original_values]
            fig.data[i].customdata = original_values
            fig.data[i].hovertemplate = fig.data[i].hovertemplate.replace("%{x}", "%{customdata}")
    if trim_y:
        for i in range(len(fig.data)):
            original_values = fig.data[i].y
            fig.data[i].y = [el if len(el) <= trim_y else f"{el[:trim_y]}..." for el in original_values]
            fig.data[i].customdata = original_values
            fig.data[i].hovertemplate = fig.data[i].hovertemplate.replace("%{y}", "%{customdata}")

    if return_fig:
        return fig
    fig.show()


def plot_area(
    df: pl.DataFrame,
    x: str,
    y: str,
    breakdown: str,
    as_share: bool = False,
    title: str = None,
    subtitle: str = None,
    x_title: str = None,
    y_title: str = None,
    legend_title: str = None,
    legend_replace: dict = None,
    return_fig: bool = False,
):
    fig = go.Figure()

    plot_y_col = y
    if as_share:
        df = df.with_columns((pl.col(y) / pl.col(y).sum()).over(pl.col(x) * 100).alias("__share_area"))
        plot_y_col = "__share_area"

    for entry, group in df.group_by(breakdown, maintain_order=True):
        fig.add_trace(go.Scatter(x=group[x], y=group[plot_y_col], name=entry, stackgroup="one"))

    add_titles(
        fig,
        title=title,
        subtitle=subtitle,
        x_title=x_title,
        y_title=y_title,
        legend_title=legend_title,
    )

    if as_share:
        fig.update_yaxes(ticksuffix="%")

    if return_fig:
        return fig
    fig.show()


# TODO: Add legend_trim behaviour
# Logic will be a bit different (single value instead of array)
def trim_labels(
    fig: go.Figure,
    x_trim: int = None,
    y_trim: int = None,
) -> go.Figure:
    """
    Reduce x and or y label length and trim with '...'.
    Original labels will then be added as customdata and the hovertemplate will
    be updated to point to them instead.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Chart whose labels will be trimmed.
    x_trim : int, default None
        Max x label size allowed before trimming.
    y_trim : int, default None
        Max y label size allowed before trimming.

    Warnings
    --------
    Changes are not inplace. The returned figure is a copy of the input to prevent repeated calls
    from needlessly increasing the customdata array.

    Raises
    ------
    ValueError
        Neither x_trim or y_trim are provided.
        Chart is empty (no data / traces).
    """

    def _append_custom_data(customdata, to_add):
        if customdata is None:
            return np.stack((to_add), axis=-1)
        else:
            return np.stack(
                [customdata[:, i] for i in range(customdata.shape[1])] + [to_add],
                axis=-1,
            )

    if not x_trim and not y_trim:
        raise ValueError("Either x_trim or y_trim must be provided")
    if len(fig.data) == 0:
        raise ValueError("Chart has no traces")

    fig = deepcopy(fig)

    for i in range(len(fig.data)):
        for selector, trim in zip(("x", "y"), (x_trim, y_trim)):
            if not trim:
                continue

            original_values = fig.data[i][selector]
            fig.data[i][selector] = [el if len(el) <= trim else f"{el[:trim]}..." for el in original_values]
            fig.data[i].customdata = _append_custom_data(fig.data[i].customdata, original_values)
            _new_index = fig.data[i].customdata.shape[-1] - 1
            fig.data[i].hovertemplate = fig.data[i].hovertemplate.replace(
                f"%{{{selector}}}", f"%{{customdata[{_new_index}]}}"
            )

    return fig


def add_titles(
    fig,
    title=None,
    subtitle=None,
    x_title=None,
    y_title=None,
    legend_title=None,
    add_log_warning=True,
):
    """
    Add provided titles, legends and axis labels in bold.
    If either x-axis and y-axis are logarithmic, a warning label will be added as well.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Chart whose labels will be updated.
    title : str, default None
        Chart title.
    title : str, default None
        Chart subtitle.
        Cannot be provided without a title.
        Added using the title argument as a separate, smaller line.
    x_title : str, default None
        X-axis title.
    y_title : str, default None
        Y-axis title.
    legend_title : str, default None
        Legend title.
        Likely won't be shown if chart only has one trace.
    add_log_warning : bool, default True
        If set to True, a warning label will be added to axes whose type is logarithmic.

    Warnings
    --------
    Changes are inplace.
    """
    chart_update_args = {}

    chart_subtitle = build_subtitle(subtitle) if subtitle else ""
    chart_update_args["title"] = bold_text(title) + chart_subtitle if title else None

    chart_x_title = x_title
    if chart_x_title and add_log_warning and fig.layout.xaxis.type == "log":
        chart_x_title += f"<br><span style='color: {layout_config['warning_color']}'>Logarithmic</span>"
    chart_update_args["xaxis_title"] = bold_text(chart_x_title) if chart_x_title else None

    chart_y_title = y_title
    if chart_y_title and add_log_warning and fig.layout.yaxis.type == "log":
        chart_y_title += f"<br><span style='color: {layout_config['warning_color']}'>Logarithmic</span>"
    chart_update_args["yaxis_title"] = bold_text(chart_y_title) if chart_y_title else None

    chart_update_args["legend_title"] = bold_text(legend_title) if legend_title else None

    fig.update_layout(**chart_update_args)
