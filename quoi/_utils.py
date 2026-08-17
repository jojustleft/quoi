import functools
import inspect
import polars as pl


def check_non_empty_data(func):
    """
    Decorator used to check if DataFrame or Series inputs are not empty.
    This is applied to *every* input of type pl.DataFrame or pl.Series.

    Raises
    ------
    ValueError
        Found DataFrame and / or Series are empty.
        No DataFrame or Series inputs (to prevent unnecessary calls).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get variable name - value pairs
        signature = inspect.signature(func)
        func_args = signature.bind(*args, **kwargs)
        func_args.apply_defaults()

        found_dt = False
        for arg, value in func_args.arguments.items():
            if isinstance(value, pl.DataFrame) or isinstance(value, pl.Series):
                found_dt = True
                if value.is_empty():
                    raise ValueError(f"Argument {arg} is empty.")

        if not found_dt:
            ValueError("No arguments of type DataFrame or Series were found.")

        return func(*args, **kwargs)

    return wrapper


def get_shared_dict_schema(dict_l: tuple | list) -> dict:
    """
    Create an empty dictionary whose key structure encompasses all provided dictionaries.

    Warnings
    --------
    - This method does not preserve iterable types that are not dictionaries: all are set to None.
    - If a given key on the input maps to a value in one dictionary and to a dictionary in another, the resulting schema will
    prioritize the dictionary due to the recursive behaviour.
    """
    schema = {}

    for d in dict_l:
        for k, v in d.items():
            if isinstance(v, dict):
                # Check against schema key (if any) and update if available
                # Notice that if there is no key, this simply copies the current value
                schema_val = schema.get(k)
                schema_input = [schema_val] if schema_val else []
                schema[k] = get_shared_dict_schema([d[k]] + schema_input)
            elif k not in schema:
                schema[k] = None

    return schema
