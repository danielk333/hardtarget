import datetime as dt

def ts_from_str(datetime_str, as_local=False):
    """
    convert from human readable string (ISO 8601 minus time zone) to timestamps (seconds since epoch)
    by default, the string is interpreded as UTC time, unless <as_local> is True,
    in which case the string is interpreted as local time.
    """
    _datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
    if not as_local:
        # Convert to UTC (assuming the datetime_str was referencing local time)
        _datetime = _datetime.replace(tzinfo=dt.timezone.utc)
    return _datetime.timestamp()


def str_from_ts(ts, as_local=False):
    """
    convert from timestamp (seconds since epoch) to human readable string (ISO 8601 minus time zone)
    returns UTC time by default - or local time if as_local is True
    """
    if as_local:
        _datetime = dt.datetime.fromtimestamp(ts)
    else:
        _datetime = dt.datetime.utcfromtimestamp(ts)
    return _datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")





