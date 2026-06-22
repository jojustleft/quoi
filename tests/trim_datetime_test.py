from datetime import datetime
import pytest
from quoi.time import trim_datetime

_default_dt = datetime(2025, 5, 1, 10, 15, 20, 203)


def test_trim_date_default():
    expected = _default_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    assert trim_datetime(_default_dt) == expected


def test_trim_to_month():
    expected = _default_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    assert trim_datetime(_default_dt, lowest_unit="month") == expected


def test_trim_to_year():
    expected = _default_dt.replace(
        month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    )

    assert trim_datetime(_default_dt, lowest_unit="year") == expected


def test_unknown_unit_error():
    with pytest.raises(ValueError):
        trim_datetime(_default_dt, lowest_unit="??")


def test_not_datetime_error():
    with pytest.raises(ValueError):
        trim_datetime(1000)
