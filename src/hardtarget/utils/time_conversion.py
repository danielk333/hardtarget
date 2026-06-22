"""Time conversion tools"""

import datetime as dt
from typing import Optional

import numpy as np

from hardtarget.types import Bounds


def time_interval_to_sample_bound(
    time_bounds: tuple[int | float, int | float],
    sample_rate: float,
    start_time: Optional[np.datetime64 | int | float | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | float | dt.datetime] = None,
    relative_time: bool = False,
) -> Bounds:
    """
    Convert a real time interval to sample indexes, each files sample always start from 0

    Args:
        time_bounds: [min, max] file epoch bounds (seconds since epoch)
        sample_rate: samples per second
        start_time (optional): start time as datetime or seconds since epoch
        end_time (optional): end time as datetime or seconds since epoch
        relative_time: If start and end should be measured from measusrement start or real time

    """

    if start_time is not None:
        if relative_time:
            if isinstance(start_time, int) or isinstance(start_time, float):
                start_sample = int(start_time * sample_rate)
            else:
                raise ValueError("Relative time is only compatible with int or float start/end")
        else:
            if isinstance(start_time, np.datetime64):
                start_sec = start_time.astype("datetime64[s]").astype("float64")
            elif isinstance(start_time, dt.datetime):
                start_sec = start_time.timestamp()
            else:
                if isinstance(start_time, int):
                    start_sec = np.datetime64(start_time, "s").astype("float64")
                else:
                    start_sec = (
                        np.datetime64(int(start_time * 1e6), "us").astype("datetime64[s]").astype("float64")
                    )

            assert start_sec >= time_bounds[0], (
                f"Start time: {str_from_ts(start_sec)} is before measurement start: {str_from_ts(time_bounds[0])}"
            )
            start_sample = int((start_sec - time_bounds[0]) * sample_rate)
    else:
        start_sample = 0

    if end_time is not None:
        if relative_time:
            if isinstance(end_time, int) or isinstance(end_time, float):
                end_sample = int(end_time * sample_rate)
            else:
                raise ValueError("Relative time is only compatible with int or float start/end")
        else:
            if isinstance(end_time, np.datetime64):
                end_sec = end_time.astype("datetime64[s]").astype("float64")
            elif isinstance(end_time, dt.datetime):
                end_sec = end_time.timestamp()
            else:
                if isinstance(end_time, int):
                    end_sec = np.datetime64(end_time, "s").astype("float64")
                else:
                    end_sec = (
                        np.datetime64(int(end_time * 1e6), "us").astype("datetime64[s]").astype("float64")
                    )

            assert end_sec <= time_bounds[1], (
                f"End time: {str_from_ts(end_sec)} s is after measurement end: {str_from_ts(time_bounds[1])}s"
            )
            end_sample = int((end_sec - time_bounds[0]) * sample_rate)
    else:
        end_sample = int((time_bounds[1] - time_bounds[0]) * sample_rate)

    return Bounds(start_sample, end_sample)


def ipp_time_to_sample(time_us: float | int, sample_rate: float) -> np.int64:
    return np.round(time_us * 1e-6 * sample_rate).astype(np.int64)


def ts_from_str(datetime_str: str, as_local: bool = False) -> float:
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


def str_from_ts(ts: float, as_local: bool = False) -> str:
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


def ts_from_index(idx: int, sample_rate: float, ts_offset_sec: float = 0) -> float:
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


def index_from_ts(ts: float, sample_rate: float, ts_offset_sec: float = 0) -> float:
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
    return (ts - ts_offset_sec) * sample_rate
