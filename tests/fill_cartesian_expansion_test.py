from datetime import datetime, timedelta
import polars as pl
import pytest
from quoi.stats import fill_cartesian_expansion


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
    expected_rows = (
        ((df["time"].max() - df["time"].min()).days + 1)
        * df["country"].n_unique()
        * df["weather"].n_unique()
    )
    print(
        (df["time"].max() - df["time"].min()).days,
        df["country"].n_unique(),
        df["weather"].n_unique(),
    )

    res = fill_cartesian_expansion(df, time_c="time", entry_l=["country", "weather"])

    assert len(res) == expected_rows


def test_entry_coverage():
    df = _get_mock_data()
    res = fill_cartesian_expansion(df, time_c="time", entry_l=["country", "weather"])

    for col in ["time", "country", "weather"]:
        value_counts = res[col].value_counts()
        assert value_counts["count"].max() == value_counts["count"].min()


def test_override_interval():
    df = _get_mock_data()
    override_start = datetime(2024, 12, 20)
    override_end = datetime(2025, 1, 20)
    expected_rows = (
        ((override_end - override_start).days + 1)
        * df["country"].n_unique()
        * df["weather"].n_unique()
    )

    res = fill_cartesian_expansion(
        df,
        time_c="time",
        entry_l=["country", "weather"],
        start_ts=override_start,
        end_ts=override_end,
    )

    assert res["time"].max() == override_end
    assert len(res) == expected_rows


def test_time_frequency():
    df = _get_mock_data()
    expected = set([timedelta(hours=1)])

    res = fill_cartesian_expansion(
        df, time_c="time", entry_l=["country", "weather"], interval="1h"
    )

    times = pl.DataFrame(res.sort("time")["time"].unique())
    times = times.with_columns(pl.col("time").diff().alias("time_diff")).drop_nulls()

    assert set(times["time_diff"].unique().to_list()) == expected


def test_default_value():
    df = _get_mock_data()
    expected = -1
    res = fill_cartesian_expansion(
        df, time_c="time", entry_l=["country", "weather"], default=expected
    )

    new_rows = res.join(df, on=["time", "country", "weather"], how="anti")

    assert set(new_rows["val"].unique().to_list()) == set([expected])


def test_redundant_call():
    df = _get_mock_data()
    res = fill_cartesian_expansion(
        df, time_c="time", entry_l=["country", "weather"]
    ).sort(["time", "country", "weather"])
    res_redundant = fill_cartesian_expansion(
        res, time_c="time", entry_l=["country", "weather"]
    ).sort(["time", "country", "weather"])

    assert res.equals(res_redundant)


def test_no_entries_error():
    df = _get_mock_data()
    with pytest.raises(ValueError):
        _ = fill_cartesian_expansion(df)
