from datetime import datetime
import polars as pl
import pytest
from quoi.stats import summary


def _get_mock_data():
    df = pl.DataFrame(
        [
            {
                "time": datetime(2025, 1, 1),
                "country": "PT",
                "weather": "sunny",
                "val": 2,
            },
            {
                "time": datetime(2025, 1, 3),
                "country": "FR",
                "weather": "cloudy",
                "val": 10,
            },
            {
                "time": datetime(2025, 1, 8),
                "country": "ES",
                "weather": "rain",
                "val": 5,
            },
            {
                "time": datetime(2025, 1, 15),
                "country": "FR",
                "weather": "foggy",
                "val": 15,
            },
        ]
    )

    return df


def test_total_rows():
    df = _get_mock_data()
    _expected = len(df.columns)

    res = summary(df)

    assert len(res) == _expected


def test_expected_columns():
    df = _get_mock_data()
    _expected = set(df.columns)

    res = set(summary(df)["column"].unique())

    assert res == _expected


def test_empty_data_error():
    df = pl.DataFrame()

    with pytest.raises(ValueError):
        _ = summary(df)
