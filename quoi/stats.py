from datetime import datetime
import polars as pl
import warnings
from typing import List

polars_allowed_types = str | int | float | bool | None
polars_data_types = pl.DataFrame | pl.Series

__all__ = [
    "summary",
    "fill_based_on",
    "fill_cartesian_expansion",
    "get_top",
    "get_value",
]


def summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate a summary table containing various metadata about each column:
    - dtype
    - number of unique values
    - number of missing values
    - percentage share of missing values (out of all rows)

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing columns whose metadata will be calculated.

    Raises
    ------
    ValueError
        DataFrame is empty.
    """
    if df.is_empty():
        raise ValueError("Received empty dataframe")

    rows = []
    for col in df.columns:
        row = {
            "column": col,
            "dtype": df[col].dtype,
            "nunique": df[col].n_unique(),
            "missing": df[col].null_count(),
            "share_missing": df[col].null_count() / df[col].len() * 100,
        }
        rows.append(row)
    return pl.DataFrame(rows)


def get_top(
    df: pl.DataFrame,
    entry_c: str,
    value_c: str,
    method: pl.Expr = pl.Expr.sum,
    calculate_share: bool = True,
) -> pl.DataFrame:
    """
    Calculate a summary of entries by provided method and order them in descending order.

    Parameters
    ----------
    df : polars.DataFrame
        Dataframe containing data with entries and values that will be aggregated.
    entry_c : str | list[str]
        Column name or list of column names containing entries to aggregate.
    value_c : str
        Column name containing values that will be aggregated.
    method : polars.expr, default polars.Expr.sum
        Aggregation operation.
    calculate_share : bool, default True
        If set to True, return results with percentage column based on sum of aggregation.
        A warning is raised if there are negative values and this calculation will be skipped.
    """

    res = df.group_by(entry_c).agg(method(pl.col(value_c))).sort(value_c, descending=True)

    if calculate_share:
        # Count aggregations are made an exception since they always range from 0 to infinity
        if res[value_c].dtype.is_numeric() and df[value_c].min() < 0 and method != pl.Expr.count:
            warnings.warn("Input contains negative values, skipping share calculation.")
        else:
            res = res.with_columns((pl.col(value_c) / pl.col(value_c).sum() * 100).alias("share"))

    return res


def get_value(
    df: pl.DataFrame,
    key_c: str,
    value_c: str,
    key: polars_allowed_types,
    default: polars_allowed_types = None,
) -> polars_allowed_types:
    """
    Obtain corresponding value from a key in a dataframe.

    Parameters
    ----------
    df : polars.DataFrame
        Dataframe containing data with entries and values that will be aggregated.
    key_c : str
        Column name containing provided key.
    value_c : str
        Column name containing value that will be returned.
    key : str
        Key to search values by.
    default : str | int | float | bool | None, default None
        Value returned if provided key is not found.

    Raises
    ------
    ValueError
        Repeated values found in key set.
    """

    # Key must be string, integer, float or boolean
    if not any([isinstance(key, t) for t in [str, int, float, bool]]):
        raise ValueError(f"Expected key of type {{str, int, float, bool}}, received {type(key)}")

    # Mapping assumes each key appears exactly once
    if set(df[key_c].value_counts()["count"]) != set([1]):
        raise ValueError("Repeated keys found in provided dataframe.")

    _mapping = dict(zip(df[key_c], df[value_c]))
    return _mapping.get(key, default)


def fill_cartesian_expansion(
    df: pl.DataFrame,
    time_c: str = None,
    entry_l: List[str] = None,
    default: polars_allowed_types = 0,
    interval: str = "1d",
    start_ts: datetime = None,
    end_ts: datetime = None,
):
    """
    Add missing entry combinations to data using cartesian product of all possible entries.
    Useful for timeseries with missing dates.

    Parameters
    ----------
    df : polars.DataFrame
        Dataframe containing data with missing entries.
    time_c : str, default None
        Column containing datetime info. If provided, will create a standalone time range based on minimum
        and maximum timestamps in provided data.
    entry_l : list[str], default None
        List of entries used to fill missing data with.
    default : str | int | float | bool | None, default None
        Value new rows will be created with.
    interval : str, default '1d'
        Delta between timestamps.
    start_ts : datetime.datetime, default None
        If provided, will override the default start time (data's lowest timestamp)
    end_ts : datetime.datetime, default None
        If provided, will override the default end time (data's highest timestamp)

    Warnings
    --------
    - Possible cartesian explosion. This method is not recommended for heavier dataframes;
    - Assumes independence between every entry column;
    - This method only considers observed entries.
    """

    if time_c is None and not entry_l:
        raise ValueError("time_c and/or entry_l must be provided.")

    join_c = []
    df_comb = pl.DataFrame()

    if time_c is not None:
        min_t = df[time_c].min() if not start_ts else start_ts
        max_t = df[time_c].max() if not end_ts else end_ts
        if isinstance(df.schema[time_c], pl.Date):
            df_comb = pl.DataFrame(pl.date_range(start=min_t, end=max_t, interval=interval, eager=True).alias(time_c))
        else:
            time_unit = df.schema[time_c].time_unit
            df_comb = pl.DataFrame(
                pl.datetime_range(
                    start=min_t,
                    end=max_t,
                    interval=interval,
                    time_unit=time_unit,
                    eager=True,
                ).alias(time_c)
            )
        join_c.append(time_c)

    if entry_l:
        df_entry = pl.DataFrame(df[entry_l[0]].unique())
        join_c.append(entry_l[0])
        for el in entry_l[1:]:
            to_add = pl.DataFrame(df[el].unique())
            df_entry = df_entry.join(to_add, how="cross")
            join_c.append(el)

        if not df_comb.is_empty():
            df_comb = df_comb.join(df_entry, how="cross")
        else:
            df_comb = df_entry

    col_order = df.columns
    res = df.join(df_comb, on=join_c, how="right")[col_order]
    res = res.fill_null(default)

    return res


def fill_based_on(df: pl.DataFrame, base: str, target: str, drop_missing: bool = False):
    """
    Fill values of target column based on rows where base column has the same value.
    This is done by obtaining a 1:1 mapping between base and target columns and making
    a join with the provided data.

    Mapping between base and target data are expected to be 1:1. A given value in base
    can only map to one unique value in target.

    Parameters
    ----------
    df : polars.DataFrame
        Dataframe containing data with missing entries.
    base : str
        Column containing keys used to map to expected values.
    target : str
        Column used to fill missing data with.
    drop_missing : bool, default False
        If set to True, values in target always missing will be dropped.

    Raises
    ------
    ValueError
        Mapping between base and target data is not 1:1.
    """

    col_order = df.columns
    mapping = (
        df.filter((~pl.col(base).is_null()) & (~pl.col(target).is_null()))
        .group_by([base, target])
        .len(name="count")
        .drop("count")
    )

    if mapping[base].value_counts()["count"].max() != 1:
        raise ValueError(f"Mapping between {base} and {target} is not 1:1.")
    if mapping.is_empty():
        raise ValueError(f"Empty mapping between {base} and {target}.")

    join_type = "full" if not drop_missing else "right"
    df = df.join(mapping, on=base, how=join_type, coalesce=True).drop(target).rename({target + "_right": target})
    return df[col_order]


def normalize(
    dt: polars_data_types,
    x: str | list = None,
    min_zero: bool = False,
    shared_scope: bool = False,
):
    """
    Perform min-max scaling on data.
    Formula: (x - x_min) / (x_max - x_min)

    Parameters
    ----------
    dt: polars.DataFrame | polars.Series
        DataFrame or Series containing data to scale.
    x: str | list, default None
        Column, or list of columns to scale.
        Only used if data is a DataFrame.
    min_zero: bool, default False
        If set to True, this will override the minimum value to be zero.
    shared_scope: bool, default false
        If set to True, specified value columns will be scaled under the same min and max values,
        thus sharing the same 0-1 range.
        Only used if data is a DataFrame.

    Warnings
    --------
    This method will preserve the format of the input.
    If data is provided as a DataFrame, it will return a DataFrame with new columns
    representing scaled data.
    If data is provided as a Series, it will return a Series.

    Raises
    ------
    ValueError
        Empty data.
        Input is not DataFrame or Series.
        Data is not numeric.
    """
    if not any([isinstance(dt, t) for t in (pl.Series, pl.DataFrame)]):
        raise ValueError(f"Expected dt as type polars.DataFrame or polars.Series, received: {type(dt)}")
    if dt.is_empty():
        raise ValueError("Received empty data")

    if isinstance(dt, pl.Series):
        if not dt.dtype.is_numeric():
            raise ValueError(f"Provided data is not numeric: {dt.dtype}")
        x_min = dt.min() if not min_zero else 0
        x_max = dt.max()

        return (dt - x_min) / (x_max - x_min)

    else:
        if not x:
            raise ValueError("x must be provided if input is of type polars.DataFrame")

        cols_to_scale = [x] if isinstance(x, str) else x
        if not all([dt[c].dtype.is_numeric() for c in cols_to_scale]):
            raise ValueError("Columns {} are not numeric".format([not dt[c].dtype.is_numeric() for c in cols_to_scale]))

        shared_x_min = None if not shared_scope else dt[cols_to_scale].min_horizontal().min()
        shared_x_max = None if not shared_scope else dt[cols_to_scale].max_horizontal().max()
        for col in cols_to_scale:
            x_min = dt[col].min() if not shared_scope else shared_x_min
            x_min = x_min if not min_zero else 0
            x_max = dt[col].max() if not shared_scope else shared_x_max
            dt = dt.with_columns(((pl.col(col) - x_min) / (x_max - x_min)).alias(col + "_normalized"))
        return dt
