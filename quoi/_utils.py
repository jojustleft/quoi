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
