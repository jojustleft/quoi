from datetime import datetime, timedelta
import polars as pl
import pytest
from quoi.stats import fill_based_on

BASE_COL = "weekday"
TARGET_COL = "weekday_name"
VALUE_COL = "value"
MISSING_VAL = 4


def _get_mock_data():
    df = pl.DataFrame(
        [
            {BASE_COL: 1, TARGET_COL: "Monday", VALUE_COL: 5},
            {BASE_COL: 2, TARGET_COL: "Tuesday", VALUE_COL: 15},
            {BASE_COL: 5, TARGET_COL: "Friday", VALUE_COL: 8},
            {BASE_COL: 3, TARGET_COL: "Wednesday", VALUE_COL: 9},
            {BASE_COL: 1, TARGET_COL: None, VALUE_COL: 50},
            {BASE_COL: 6, TARGET_COL: "Saturday", VALUE_COL: 90},
            {BASE_COL: MISSING_VAL, TARGET_COL: None, VALUE_COL: 90},
        ]
    )

    return df


def test_same_shape():
    df = _get_mock_data()
    expected_shape = df.shape

    res = fill_based_on(df, base=BASE_COL, target=TARGET_COL)
    assert res.shape == expected_shape


def test_same_value_sum():
    df = _get_mock_data()
    expected_sum = df[VALUE_COL].sum()

    res = fill_based_on(df, base=BASE_COL, target=TARGET_COL)
    assert res[VALUE_COL].sum() == expected_sum


def test_always_missing_dropped():
    df = _get_mock_data()

    res = fill_based_on(df, base=BASE_COL, target=TARGET_COL, drop_missing=True)
    assert res.filter(res[TARGET_COL].is_null()).is_empty()


def test_always_missing_kept():
    df = _get_mock_data()

    res = fill_based_on(df, base=BASE_COL, target=TARGET_COL)
    assert set(res.filter(res[TARGET_COL].is_null())[BASE_COL].unique()) == set([MISSING_VAL])


def test_not_1_1_mapping_error():
    df = _get_mock_data()
    df = df.extend(pl.DataFrame({BASE_COL: 1, TARGET_COL: "Friday", VALUE_COL: 20}))

    with pytest.raises(ValueError):
        _ = fill_based_on(df, base=BASE_COL, target=TARGET_COL)


def test_no_entries_error():
    df = pl.DataFrame(schema=[BASE_COL, TARGET_COL, VALUE_COL])

    with pytest.raises(ValueError):
        _ = fill_based_on(df, base=BASE_COL, target=TARGET_COL)
