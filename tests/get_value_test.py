import polars as pl
import pytest
from quoi.stats import get_value

_key_c = 'country'
_value_c = 'name'


def _get_mock_data():
    df = pl.DataFrame([
        {'country': 'PT', 'name': 'Portugal'},
        {'country': 'FR', 'name': 'France'},
        {'country': 'DE', 'name': 'Germany'},
        {'country': 'JP', 'name': 'Japan'},
    ])

    return df

def test_missing_value():
    df = _get_mock_data()
    
    assert get_value(df, key_c=_key_c, value_c= _value_c,
                     key='??') is None

def test_missing_value_default():
    df = _get_mock_data()
    _expected = -1
    
    assert get_value(df, key_c=_key_c, value_c= _value_c,
                     key='??',
                     default=_expected) == _expected


def test_repeated_key_error():
    df = _get_mock_data()
    # Duplicate data
    df = df.extend(df)

    with pytest.raises(ValueError):
        get_value(df, key_c=_key_c, value_c= _value_c, key='PT')

def test_existing_value():
    df = _get_mock_data()
    
    assert get_value(df, key_c=_key_c, value_c= _value_c, key='PT') == 'Portugal'
