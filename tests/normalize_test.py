import numpy as np
import polars as pl
import pytest
from quoi.stats import normalize

SCALE_COLS = ["standard", "upper_bound"]


def _get_mock_data():
    df = pl.DataFrame({"standard": list(range(50)), "upper_bound": list(range(50, 150, 2))})

    return df


def test_return_dataframe():
    df = _get_mock_data()
    res = normalize(df, x=SCALE_COLS)

    assert isinstance(res, pl.DataFrame)


def test_return_series():
    df = _get_mock_data()
    res = normalize(df["standard"])

    assert isinstance(res, pl.Series)


def test_added_columns():
    df = _get_mock_data()
    res = normalize(df, x=SCALE_COLS)

    expected = set([col + "_normalized" for col in SCALE_COLS])
    assert not expected.difference(set(res.columns))


def test_independent_scope():
    df = _get_mock_data()
    res = normalize(df, x=SCALE_COLS)

    for col in SCALE_COLS:
        assert np.isclose(res[col + "_normalized"].min(), 0)
        assert np.isclose(res[col + "_normalized"].max(), 1)


def test_shared_scope():
    df = _get_mock_data()
    res = normalize(df, x=SCALE_COLS, shared_scope=True)

    assert not np.isclose(res["standard_normalized"].max(), 1)
    assert np.isclose(res["upper_bound_normalized"].max(), 1)


def test_min_zero():
    df = _get_mock_data()
    res = normalize(df, x=SCALE_COLS, min_zero=True)

    for col in SCALE_COLS:
        if df[col].min() > 0:
            assert not np.isclose(res[col + "_normalized"].min(), 0)


def test_empty_data_error():
    df = pl.DataFrame()

    with pytest.raises(ValueError):
        _ = normalize(df, x=SCALE_COLS)


def test_unsupported_type_error():
    df = list(range(0, 10))

    with pytest.raises(ValueError):
        _ = normalize(df)


def test_non_numeric_series_error():
    df = pl.Series(["PT", "FR", "ES", "JP"])

    with pytest.raises(ValueError):
        _ = normalize(df)
