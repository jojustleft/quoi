import polars as pl
from quoi.stats import z_score
from collections import namedtuple
from typing import NamedTuple

from quoi.viz import plot_anomalies, add_titles

AnomalyResults = namedtuple("AnomalyResults", ["series", "fig"])


def z_score_anomalies(df: pl.DataFrame, x: "str", y: "str", threshold: int = 3) -> NamedTuple:
    """
    Perform anomaly detection based on Z-score.

    Using this method, an obvervation is considered anomalous if its deviation from the mean is above a given number of
    standard deviations (threshold).

    Threshold is 3 standard deviations by default as it represents a moderate sensitivity to outliers. This value is
    inversely related to anomaly sensitivity: decreasing the threshold will increase anomaly sensitivity and
    vice-versa.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing sample data.
    x : str
        X-axis column, normaly index / time data.
    y : str
        Y-axis column with data that will be checked for anomalies.
    threshold : int, default 3
        Amount of standard deviations that make up the boundary separating outliers.

    Warnings
    --------
    This method assumes the data follows a normal distribution. To check this, consider the following approaches:
     - plot an histogram of the distribution;
     - quoi.viz.qq_plot for visualizing the Q-Q plot;
     - Shapiro-Wilk test.
    """

    anomaly_col_name = "__is_z_score_anomaly"

    df = df.with_columns(z_score=z_score(df[y])).with_columns(
        (pl.col("z_score").abs() > threshold).alias(anomaly_col_name)
    )

    mean, standard_deviation = df[y].mean(), df[y].std()
    upper_bound = mean + (standard_deviation * threshold)
    lower_bound = mean - (standard_deviation * threshold)

    fig = plot_anomalies(df, x=x, y=y, anomaly_col=anomaly_col_name, lower_bound=lower_bound, upper_bound=upper_bound)

    add_titles(fig, title="Detected Z-score anomalies", subtitle=f"Using a threshold of {threshold}")

    return AnomalyResults(series=df[anomaly_col_name], fig=fig)


def iqr_anomalies(df: pl.DataFrame, x: "str", y: "str"):
    """
    Perform anomaly detection based on Interquartile range.

    Using this method, an obvervation is considered anomalous if if falls below Q1 - (1.5 * IQR)
    or above Q3 + (1.5 * IQR). With IQR = Q3 - Q1.

    Since this method is percentile-based, it is less sensitive to outliers than Z-score.
    This method does not assume the data follows a normal distribution.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing sample data.
    x : str
        X-axis column, normaly index / time data.
    y : str
        Y-axis column with data that will be checked for anomalies.
    """

    anomaly_col_name = "__is_iqr_anomaly"

    q1, q3 = df[y].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    df = df.with_columns(((pl.col(y) < lower_bound) | (pl.col(y) > upper_bound)).alias(anomaly_col_name))

    fig = plot_anomalies(df, x=x, y=y, anomaly_col=anomaly_col_name, lower_bound=lower_bound, upper_bound=upper_bound)

    add_titles(fig, title="Detected IQR anomalies")

    return AnomalyResults(series=df[anomaly_col_name], fig=fig)
