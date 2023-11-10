from pathlib import Path
import datetime
import configparser
import numpy as np
import digital_rf as drf

"""
Utility function for accessing Hardtarget DRF
"""

STRING_PROPS = ["name", "version", "rx_channel", "tx_channel"]
FLOAT_PROPS = [
    "tx_pulse_length",
    "file_secs",
    "doppler_sign",
    "rx_start",
    "rx_end",
    "tx_start",
    "tx_end",
    "cal_on",
    "cal_off",
    "radar_frequency",
]
INT_PROPS = ["sample_rate", "ipp"]
BOOL_PROPS = ["round_trip_range"]
SECTIONS = ["Experiment"]


####################################################################
# LOAD HARDTARGET DRF PARAMS
####################################################################

def load_hardtarget_drf(dstdir):
    reader = load_hardtarget_drf_reader(dstdir)
    params = load_hardtarget_drf_params(dstdir)
    return reader, params


def load_hardtarget_drf_reader(dstdir):
    return drf.DigitalRFReader(str(dstdir))


def load_hardtarget_drf_params(dstdir):
    """
    Returns (reader, meta)
    - reader is a Digital_rf Reader object
    - meta is a dict with metadata
    """
    dstdir = Path(dstdir)

    # metadata file
    meta = configparser.ConfigParser()
    meta.read(dstdir / "metadata.ini")

    # parse meta data
    d = {}
    for section in SECTIONS:
        for prop in meta[section].keys():
            if prop in INT_PROPS:
                d[prop] = meta.getint(section, prop)
            elif prop in BOOL_PROPS:
                d[prop] = meta.getboolean(section, prop)
            elif prop in FLOAT_PROPS:
                d[prop] = meta.getfloat(section, prop)
            elif prop in STRING_PROPS:
                d[prop] = meta.get(section, prop).strip("'").strip('"')
            else:
                d[prop] = meta.get(section, prop)
    return d


# TODO
# Could also support default parameters if the metadata file does not exist


####################################################################
# TIME INTERVAL TO SAMPEL BOUNDS
####################################################################


def time_interval_to_samples(start_time, end_time, bounds, sample_rate, relative_time=False):
    """Convert a time interval to samples while checking bounds, allows for times relative bounds
    """
    interval = [x for x in bounds]

    if relative_time:
        start_time = float(start_time)
        end_time = float(end_time)

    if start_time is not None:
        if relative_time:
            _b0 = bounds[0] + start_time * sample_rate
            unix_t0 = _b0 / sample_rate
        else:
            if not isinstance(start_time, np.datetime64):
                dt64_t0 = np.datetime64(start_time)
            else:
                dt64_t0 = start_time
            unix_t0 = dt64_t0.astype("datetime64[s]").astype("int64")
            _b0 = unix_t0 * sample_rate

        dt0 = datetime.datetime.utcfromtimestamp(bounds[0]/sample_rate)
        dts = datetime.datetime.utcfromtimestamp(unix_t0)
        assert _b0 >= bounds[0], f"Given start time '{dts}' is before input data start '{dt0}'"
        interval[0] = _b0

    if end_time is not None:
        if relative_time:
            _b1 = bounds[0] + end_time * sample_rate
            unix_t1 = _b1 / sample_rate
        else:
            if not isinstance(end_time, np.datetime64):
                dt64_t1 = np.datetime64(end_time)
            else:
                dt64_t1 = end_time
            unix_t1 = dt64_t1.astype("datetime64[s]").astype("int64")
            _b1 = int(unix_t1 * sample_rate)
        dt1 = datetime.datetime.utcfromtimestamp(bounds[1]/sample_rate)
        dte = datetime.datetime.utcfromtimestamp(unix_t1)
        assert _b1 <= bounds[1], f"Given end time '{dte}' is after input data end '{dt1}'"
        interval[1] = _b1

    return interval


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    import sys
    import pprint

    dstdir = sys.argv[1]
    d = load_hardtarget_drf_params(dstdir)
    pprint.pprint(d)
