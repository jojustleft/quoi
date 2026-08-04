from dataclasses import dataclass
import plotly.graph_objects as go
import polars as pl
from quoi.stats import z_score
from typing import Protocol, Optional

from quoi.viz import plot_anomalies, add_titles

__all__ = ["z_score_anomalies", "iqr_anomalies"]


class AnomalyBoundary(Protocol):
    def detect_anomalies(self, series: pl.Series) -> bool: ...


@dataclass
# Handles both Z-score and IQR for global windows
class ThresholdBoundary(AnomalyBoundary):
    lower: float
    upper: float

    def detect_anomalies(self, series) -> pl.Series:
        return ~series.is_between(self.lower, self.upper)


@dataclass
# Rolling windows counterpart
class RollingBoundary(AnomalyBoundary):
    lower: pl.Series
    upper: pl.Series

    def detect_anomalies(self, series) -> pl.Series:
        return ~series.is_between(self.lower, self.upper)


@dataclass(frozen=True, order=True)
class AnomalyResults:
    is_anomaly: pl.Series
    boundary: AnomalyBoundary
    # IQR, for example, has no "score"
    score: Optional[pl.Series] = None
    # Only used if 'x' is provided in each method
    fig: Optional[go.Figure()] = None


def z_score_anomalies(
    df: pl.DataFrame, y: str, x: str = None, threshold: int = 3, rolling_window: int = None
) -> AnomalyResults:
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
    x : str, default None
        X-axis column, normally index / time data.
        If not provided, the chart visualization will be skipped.
    y : str
        Y-axis column with data that will be checked for anomalies.
    threshold : int, default 3
        Amount of standard deviations that make up the boundary separating outliers.
    rolling_window : int, default None
        If provided, will calculate a rolling equivalent of this method, instead of considering the entire data.
        Useful for data that register gradual changes.

    See also
    --------
    quoi.stats.qq_plot : visualize the Quantile-Quantile plot of a distribution against a normal distribution
        (used to check for normality).
    quoi.outliers.iqr_anomalies : anomaly detection based on Interquartile range (less sensitive to outliers,
        does not assume normal distribution).

    Warnings
    --------
    This method assumes the data follows a normal distribution. To check this, consider the following approaches:
     - plot an histogram of the distribution;
     - quoi.stats.qq_plot for visualizing the Q-Q plot;
     - Shapiro-Wilk test.

    When using rolling windows, please ensure the data is sorted.
    """

    anomaly_col_name = "__is_z_score_anomaly"
    score_col_name = "__z_score"
    upper_bound_col_name = "__upper_bound"
    lower_bound_col_name = "__lower_bound"

    if not rolling_window:
        df = df.with_columns(__z_score=z_score(df[y])).with_columns(
            (pl.col(score_col_name).abs() > threshold).alias(anomaly_col_name)
        )
        mean, standard_deviation = df[y].mean(), df[y].std()
        upper_bound = mean + (standard_deviation * threshold)
        lower_bound = mean - (standard_deviation * threshold)
        df = df.with_columns(
            pl.lit(upper_bound).alias(upper_bound_col_name),
            pl.lit(lower_bound).alias(lower_bound_col_name),
        )
        boundaries = ThresholdBoundary(lower=lower_bound, upper=upper_bound)

    else:
        # Since Polars creates columns concurrently, we must separate columns calculated sequentially in different method calls
        df = (
            df.with_columns(
                pl.col(y).rolling_mean(rolling_window).alias("__mean"),
                pl.col(y).rolling_std(rolling_window).alias("__std"),
            )
            .with_columns(((pl.col(y) - pl.col("__mean")) / pl.col("__std")).alias(score_col_name))
            .with_columns(
                (pl.col(score_col_name).abs() > threshold).alias(anomaly_col_name),
                (pl.col("__mean") + (pl.col("__std") * threshold)).alias(upper_bound_col_name),
                (pl.col("__mean") - (pl.col("__std") * threshold)).alias(lower_bound_col_name),
            )
        )
        boundaries = RollingBoundary(lower=df[lower_bound_col_name], upper=df[upper_bound_col_name])

    fig = None
    if x:
        fig = plot_anomalies(
            df,
            x=x,
            y=y,
            anomaly_col=anomaly_col_name,
            lower_bound=lower_bound_col_name,
            upper_bound=upper_bound_col_name,
        )
        add_titles(fig, title="Detected Z-score anomalies", subtitle=f"Using a threshold of {threshold}")

    return AnomalyResults(is_anomaly=df[anomaly_col_name], boundary=boundaries, score=df[score_col_name], fig=fig)


def iqr_anomalies(df: pl.DataFrame, y: str, x: str = None, rolling_window: int = None) -> AnomalyResults:
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
        X-axis column, normally index / time data.
        If not provided, the chart visualization will be skipped.
    y : str
        Y-axis column with data that will be checked for anomalies.
    rolling_window : int, default None
        If provided, will calculate a rolling equivalent of this method, instead of considering the entire data.
        Useful for data that register gradual changes.

    See also
    --------
    quoi.outliers.z_score_anomalies : anomaly detection based on Z-score (more apt for data following normal distributions).

    Warnings
    --------
    When using rolling windows, please ensure the data is sorted.
    """

    anomaly_col_name = "__is_iqr_anomaly"
    upper_bound_col_name = "__upper_bound"
    lower_bound_col_name = "__lower_bound"
    iqr_col_name = "__iqr"
    q1_col_name, q3_col_name = "__q1", "__q3"

    if not rolling_window:
        q1, q3 = df[y].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        df = df.with_columns(
            pl.lit(upper_bound).alias(upper_bound_col_name),
            pl.lit(lower_bound).alias(lower_bound_col_name),
        )
        boundaries = ThresholdBoundary(lower=lower_bound, upper=upper_bound)

    else:
        df = (
            df.with_columns(
                pl.col(y).rolling_quantile(quantile=0.25, window_size=rolling_window).alias(q1_col_name),
                pl.col(y).rolling_quantile(quantile=0.75, window_size=rolling_window).alias(q3_col_name),
            )
            .with_columns(
                (pl.col(q3_col_name) - pl.col(q1_col_name)).alias(iqr_col_name),
            )
            .with_columns(
                (pl.col(q1_col_name) - (1.5 * pl.col(iqr_col_name))).alias(lower_bound_col_name),
                (pl.col(q3_col_name) + (1.5 * pl.col(iqr_col_name))).alias(upper_bound_col_name),
            )
        )
        boundaries = RollingBoundary(lower=df[lower_bound_col_name], upper=df[upper_bound_col_name])

    df = df.with_columns(
        ((pl.col(y) < pl.col(lower_bound_col_name)) | (pl.col(y) > pl.col(upper_bound_col_name))).alias(
            anomaly_col_name
        ),
    )

    fig = None
    if x:
        fig = plot_anomalies(
            df,
            x=x,
            y=y,
            anomaly_col=anomaly_col_name,
            lower_bound=lower_bound_col_name,
            upper_bound=upper_bound_col_name,
        )
        add_titles(fig, title="Detected IQR anomalies")

    return AnomalyResults(is_anomaly=df[anomaly_col_name], boundary=boundaries, fig=fig)
