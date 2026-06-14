import polars as pl
import warnings

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