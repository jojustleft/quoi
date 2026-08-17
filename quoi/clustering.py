import numpy as np
import plotly.graph_objects as go
import polars as pl
from sklearn.decomposition import PCA

from quoi._utils import check_non_empty_data
from quoi.viz import add_titles


@check_non_empty_data
def scree_plot(df: pl.DataFrame, total_components: int, return_fig: bool = False) -> go.Figure:
    pca = PCA(n_components=total_components)
    pca.fit(df.to_numpy())

    index = np.arange(total_components) + 1
    val = pca.explained_variance_ratio_ * 100
    cumulative = val.cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=val, name="Variance ratio"))
    fig.add_trace(go.Scatter(x=index, y=cumulative, name="Cumulative"))

    add_titles(fig, title="Scree plot", x_title="Total components", y_title="Ratio")

    fig.update_layout(xaxis_tickmode="linear", yaxis_ticksuffix="%")

    if return_fig:
        return fig
    else:
        fig.show()


@check_non_empty_data
def hopkins_statistic(df: pl.DataFrame, sample_ratio: float = 0.05) -> float:
    # Reference: https://en.wikipedia.org/wiki/Hopkins_statistic
    n_rows, n_cols = df.shape

    # X: Random sample from dataset
    m = round(n_rows * sample_ratio)
    x_sample = df.sample(m, with_replacement=False)

    # Y: Random uniform points within each dimension's range
    mins = df.min().rows()[0]
    maxs = df.max().rows()[0]
    y_cols = []
    for i in range(n_cols):
        y_cols.append(pl.Series(np.random.uniform(mins[i], maxs[i], m)).alias(df.columns[i]))
    y_sample = pl.DataFrame(y_cols)

    u_d = []
    w_d = []
    for i in range(m):
        # For both x and y samples, calculate the euclidean distance to the entire dataset
        # and pick the smallest non-zero distance
        row_x = x_sample.row(i, named=True)
        curr_w = (
            x_sample.with_columns(
                pl.sum_horizontal((pl.col(c) - row_x[c]).pow(2) for c in x_sample.columns).sqrt().alias("eucl_dist")
            )
            .sort("eucl_dist")
            .filter(pl.col("eucl_dist") != 0)["eucl_dist"]
            .min()
        )
        w_d.append(curr_w)

        row_y = y_sample.row(i, named=True)
        curr_u = (
            y_sample.with_columns(
                pl.sum_horizontal((pl.col(c) - row_y[c]).pow(2) for c in y_sample.columns).sqrt().alias("eucl_dist")
            )
            .sort("eucl_dist")
            .filter(pl.col("eucl_dist") != 0)["eucl_dist"]
            .min()
        )
        u_d.append(curr_u)

    h = sum(u_d) / (sum(u_d) + sum(w_d))
    return h
