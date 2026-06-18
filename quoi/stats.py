from datetime import datetime
import polars as pl
import warnings
from typing import List

polars_allowed_types = str | int | float | bool | None

def get_top(df: pl.DataFrame,
            entry_c: str,
            value_c: str,
            method: pl.Expr = pl.Expr.sum,
            calculate_share : bool = True) -> pl.DataFrame:
    '''
    Calculate a summary of entries by provided method and order them in descending order.

    Parameters
    ----------
    df : polars.DataFrame
        Dataframe containing data with entries and values that will be aggregated.
    entry_c : str
        Column name containing entries to aggregate.
    value_c : str
        Column name containing values that will be aggregated.
    method : polars.expr, default polars.Expr.sum
        Aggregation operation.
    calculate_share : bool, default True
        If set to True, return results with percentage column based on sum of aggregation.
        A warning is raised if there are negative values and this calculation will be skipped.

    Returns
    -------
    polars.DataFrame
    '''

    res = df.group_by(entry_c).agg(method(pl.col(value_c))).sort(value_c, descending=True)

    if calculate_share:
        if res[value_c].dtype.is_numeric() and df[value_c].min() < 0:
            warnings.warn('Input contains negative values, skipping share calculation.')
        else:
            res = res.with_columns((pl.col(value_c) / pl.col(value_c).sum() * 100).alias('share'))

    return res


def get_value(df: pl.DataFrame,
              key_c: str,
              value_c: str,
              key: polars_allowed_types,
              default: polars_allowed_types = None) -> polars_allowed_types:
    '''
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
        ValueError : Repeated values found in key set.
    '''
    # TODO: Check first if key is within expected key column data types, fail if not
    
    # Mapping assumes each key appears exactly once 
    if set(df[key_c].value_counts()['count']) != set([1]):
        raise ValueError('Repeated keys found in provided dataframe.')

    _mapping = dict(zip(df[key_c], df[value_c]))
    return _mapping.get(key, default)

def fill_cartesian_expansion(df: pl.DataFrame,
                             time_c: str = None,
                             entry_l: List[str] = None,
                             default: polars_allowed_types = 0,
                             interval: str = '1d',
                             start_ts: datetime = None,
                             end_ts: datetime = None):
    '''
    Add missing entry combinations to data using cartesian product of all possible entries.
    Useful for timeseries with missing dates.

    Caveats:
     - Possible cartesian explosion. This method is not recommended for heavier dataframes;
     - Assumes independence between every entry column;
     - This method only considers observed entries.

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

    Returns
    -------
    polars.DataFrame
    '''

    if time_c is None and not entry_l:
        raise ValueError('time_c and/or entry_l must be provided.')

    join_c = []
    df_comb = pl.DataFrame()
    
    if time_c is not None:
        min_t = df[time_c].min() if not start_ts else start_ts
        max_t = df[time_c].max() if not end_ts else end_ts
        if isinstance(df.schema[time_c], pl.Date):
            df_comb = pl.DataFrame(pl.date_range(start=min_t,
                                                 end=max_t,
                                                 interval=interval,
                                                 eager=True).alias(time_c))
        else:
            time_unit = df.schema[time_c].time_unit
            df_comb = pl.DataFrame(pl.datetime_range(start=min_t,
                                                     end=max_t,
                                                     interval=interval,
                                                     time_unit=time_unit,
                                                     eager=True).alias(time_c))
        join_c.append(time_c)

    if entry_l:
        df_entry = pl.DataFrame(df[entry_l[0]].unique())
        join_c.append(entry_l[0])
        for el in entry_l[1:]:
            to_add = pl.DataFrame(df[el].unique())
            df_entry = df_entry.join(to_add, how='cross')
            join_c.append(el)

        if not df_comb.is_empty():
            df_comb = df_comb.join(df_entry, how='cross')
        else:
            df_comb = df_entry
    
    col_order = df.columns
    res = df.join(df_comb, on=join_c, how='right')[col_order]
    res = res.fill_null(default)

    return res
