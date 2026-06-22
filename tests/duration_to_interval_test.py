import pytest
from quoi.time import duration_to_interval

_default_duration = 60
_default_res = dict(days=0, hours=0, minutes=0, seconds=0, milliseconds=0)


def test_seconds_duration_default():
    expected = _default_res
    expected.update(minutes=1)

    assert duration_to_interval(_default_duration) == expected


def test_full_datetime_to_interval_seconds():
    days = 10
    hours = 5
    minutes = 45
    seconds = 20
    expected = _default_res
    expected.update(days=days, hours=hours, minutes=minutes, seconds=seconds)

    duration = (days * 60 * 60 * 24) + (hours * 60 * 60) + (minutes * 60) + seconds

    assert duration_to_interval(duration) == expected


def test_full_datetime_to_interval_milliseconds():
    days = 10
    hours = 5
    minutes = 45
    seconds = 20
    milliseconds = 850
    expected = _default_res
    expected.update(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )

    duration = (
        (days * 60 * 60 * 24 * 1000)
        + (hours * 60 * 60 * 1000)
        + (minutes * 60 * 1000)
        + (seconds * 1000)
        + milliseconds
    )

    assert duration_to_interval(duration, units="ms") == expected


def test_unknown_unit_type_error():
    with pytest.raises(ValueError):
        duration_to_interval(_default_duration, units="??")


def test_non_integer_duration_error():
    with pytest.raises(ValueError):
        duration_to_interval("500")
