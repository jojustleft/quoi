import polars as pl
import plotly.graph_objects as go
import plotly.io as pio

layout_config = dict(
    text="#201B23",
    background="#FCF6F5",
    grid="#ebe1df",
    entries=["#5C825B", "#D5834F", "#4E76B2", "#904A58", "#C0E0DE"],
    serif_font="Georgia",
    default_font="Helvetica",
)

layout = dict(
    layout=go.Layout(
        paper_bgcolor=layout_config["background"],
        plot_bgcolor=layout_config["background"],
        title=dict(
            font=dict(
                family=layout_config["serif_font"], size=22, color=layout_config["text"]
            ),
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
    return f"<br><span style='font-family: {layout_config['default_font']}; font-size: 20px; color:#4a464d'><sup>{s}</sup></span>"


def bold_text(s):
    return f"<b>{s}</b>"


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

    fig = go.Figure()

    if breakdown is not None:
        for entry, group in df.group_by(breakdown):
            curr_name = (
                entry[0]
                if not legend_replace
                else legend_replace.get(entry[0], entry[0])
            )
            fig.add_trace(go.Scatter(x=group[x], y=group[y], name=curr_name))

    else:
        cols_to_plot = [y] if isinstance(y, str) else y
        for el in cols_to_plot:
            curr_name = el if not legend_replace else legend_replace.get(el, el)
            fig.add_trace(go.Scatter(x=df[x], y=df[el], name=curr_name))

    chart_subtitle = build_subtitle(subtitle) if subtitle else ""
    chart_title = bold_text(title) + chart_subtitle if title else None
    chart_x_title = bold_text(x_title) if x_title else None
    chart_y_title = bold_text(y_title) if y_title else None
    chart_legend_title = bold_text(legend_title) if legend_title else None

    fig.update_layout(
        title=chart_title,
        xaxis_title=chart_x_title,
        yaxis_title=chart_y_title,
        legend_title=chart_legend_title,
    )

    if hovertemplate == "default":
        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "%{x}",
                    ("<b>" + legend_title + "</b>: %{fullData.name}")
                    if legend_title
                    else "",
                    "<b>%{yaxis.title.text}:</b> %{y}<extra></extra>",
                ]
            ).replace("<br><br>", "<br>")
        )

    if return_fig:
        return fig
    fig.show()
