from datetime import datetime
from collections import OrderedDict


def duration_to_interval(duration: int, units='s'):
    '''
    Convert a duration value to a human readable format.

    Parameters
    ----------
    d : int
        Number representing duration to convert.
    units : str, default 's'
        Time unit provided duration is in. Defaults to seconds. Possible values are:
         - 's': seconds
         - 'ms': milliseconds

    Returns
    -------
    dict
    '''
    if not isinstance(duration, int):
        raise ValueError("Expected int for 'duration', received {}".format(type(duration)))
    if units not in ('s', 'ms'):
        raise ValueError('Unknown unit type: {}'.format(units))

    milliseconds = 0
    if units == 'ms':
        milliseconds = duration % 1000
        duration = (duration - milliseconds) // 1000
    
    days = duration // (60 * 60 * 24)
    duration -= days * 60 * 60 * 24
    hours = duration // (60 * 60)
    duration -= hours * 60 * 60
    minutes = duration // 60
    duration -= minutes * 60
    seconds = duration
    return dict(days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                milliseconds=milliseconds)

def trim_datetime(dt, lowest_unit='day'):
    '''
    Remove (set to 0) units smaller than provided lowest unit.
    Useful for rounding current timestamp to date only.

    Parameters
    ----------
    dt : datetime.datetime
        Timestamp to round down.
    lowest_unit : str, default 'd'
        Lowest time unit that won't be removed. Possible values are: ('year', 'month', 
        'day', 'hour', 'minute', 'second')

    Returns
    -------
    datetime.datetime
        Datetime with provided units below lowest unit set to their lowest values (0 or 1).

    Raises
    ------
    ValueError
        Provided datetime not of type datetime.datetime.
        Unknown time unit type for lowest unit.
    '''
    # There's likely a better way to do this
    # Keeps order of units and their default values
    units = OrderedDict(
        year=1,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    if not isinstance(dt, datetime):
        raise ValueError("Expected datetime.datetime for 'dt', received {}".format(type(dt)))
    if lowest_unit not in (list(units.keys())[:-1]):
        raise ValueError('Unknown unit type: {}'.format(units))

    replace = False
    replace_kwargs = {}
    for u in units.keys():
        if u == lowest_unit:
            replace = True
            continue
        
        elif replace:
            replace_kwargs[u] = units[u]

    return dt.replace(**replace_kwargs)
