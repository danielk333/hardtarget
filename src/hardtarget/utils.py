import datetime as dt


####################################################################
# CONVERSIONS BETWEEN TIMESTAMP AND HUMAN READABLE STRING
####################################################################



def ts_from_str(datetime_str, as_local=False):
    """
    Convert from human-readable string (ISO 8601 without a time zone) to a timestamp (seconds since epoch).
    
    By default, the string is interpreted as UTC time unless <as_local> is True, in which case
    the string is interpreted as local time.
    """
    # Parse the string into a naive datetime object
    _datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
    
    if as_local:
        # Make it timezone-aware as local time
        _datetime = _datetime.astimezone()
    else:
        # Make it timezone-aware as UTC
        _datetime = _datetime.replace(tzinfo=dt.timezone.utc)
    
    # Return the timestamp
    return _datetime.timestamp()


def str_from_ts(ts, as_local=False):
    """
    Convert from a timestamp (seconds since epoch) to a human-readable string (ISO 8601 without a time zone).
    
    Returns UTC time by default, or local time if <as_local> is True.
    """
    if as_local:
        # Convert to local time (timezone-aware)
        _datetime = dt.datetime.fromtimestamp(ts).astimezone()
    else:
        # Convert to UTC (timezone-aware)
        _datetime = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    
    # Format the datetime as a string
    return _datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")



####################################################################
# CONVERSIONS BETWEEN TIMESTAMPS AND SAMPLE INDEXES
####################################################################


def ts_from_index(idx, sample_rate, ts_offset_sec=0):
    """
    convert from sample idx to timestamp

    Params
    ------

    idx: int
        sample index (first sample is index 0)
    sample_rate: Hz
        samples per seconds
    ts_offset_sec: float
        timestamp in seconds since Epoch (1970:01:10T00:00:00)
        ts_offset is the timestamp corresponding to index 0,
        by default this is 0, implying that indexing starts at Epoch (1970:01:10T00:00:00)

    Returns
    -------
    float:
        timestamp corresponding to given sample index

    """
    return (idx / float(sample_rate)) + ts_offset_sec


def index_from_ts(ts, sample_rate, ts_offset_sec=0):
    """
    convert from timestamp to sample index

    Params
    ------

    ts: float
        timestamp in seconds from Epoch (1970:01:10T00:00:00)
    sample_rate: Hz
        samples per seconds
    ts_offset_sec: float
        timestamp in seconds since Epoch (1970:01:10T00:00:00)
        ts_offset is the timestamp corresponding to index 0,
        by default this is 0, implying that indexing starts at Epoch (1970:01:10T00:00:00)

    Returns
    -------
    float:
        sample index (first sample is index 0)
    """
    return (ts-ts_offset_sec) * sample_rate
