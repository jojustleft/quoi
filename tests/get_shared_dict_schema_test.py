import pytest
from quoi._utils import get_shared_dict_schema

def simple_dicts():
    d1 = {'field_1_1': 123, 'field_1_2': '...', 'field_shared': 100}
    d2 = {'field_2_1': 150, 'field_shared': 250}
    return d1, d2

def nested_dicts():
    d1 = {'field_1_1': 123, 'field_1_2': '...', 
          'field_nested_1': {'field_1_3': True},
          'field_nested_shared': {'field_shared_1': 1}
    }
    d2 = {'field_2_1': 150, 'field_nested_shared': {'field_2_2': 384, 'field_2_empty': None}}
    return d1, d2

def is_empty(d):
    for k, v in d.items():
        if isinstance(v, dict):
            if not is_empty(v):
                return False
        elif v:
            return False
    return True


def test_shared_simple():
    d1, d2 = simple_dicts()
    res = get_shared_dict_schema([d1, d2])

    assert set(d1.keys()).union(d2.keys()) == set(res.keys())


def test_all_nested():
    d1, d2 = nested_dicts()
    res = get_shared_dict_schema([d1, d2])

    nested_keys_res = set([k for k, v in res.items() if isinstance(v, dict)])
    nested_keys_d1 = set([k for k, v in d1.items() if isinstance(v, dict)])
    nested_keys_d2 = set([k for k, v in d2.items() if isinstance(v, dict)])

    assert set(nested_keys_res) == set(nested_keys_d1).union(nested_keys_d2)


def test_schema_empty_simple():
    d1, d2 = simple_dicts()
    res = get_shared_dict_schema([d1, d2])

    assert is_empty(res)


def test_schema_empty_nested():
    d1, d2 = nested_dicts()
    res = get_shared_dict_schema([d1, d2])

    assert is_empty(res)