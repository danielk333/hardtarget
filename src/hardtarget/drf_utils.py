from pathlib import Path
import configparser
import digital_rf as drf

"""
Utility function for accessing Hardtarget DRF
"""

STRING_PROPS = ["name", "version", "rx_channel", "tx_channel"]
FLOAT_PROPS = [
    "file_secs", "pulse_length", "doppler_sign", 
    "rx_start", "rx_end", "tx_start", "tx_end",
    "cal_on", "cal_off", "frequency"
]
INT_PROPS = ["sample_rate", "ipp"]
BOOL_PROPS = ["round_trip_range"]
SECTIONS = ["Experiment"]


def get_hardtarget_drf(dstdir):
    """
    Returns (reader, meta)
    - reader is a Digital_rf Reader object
    - meta is a dict with metadata
    """
    dstdir = Path(dstdir)

    # digital rf reader
    reader = drf.DigitalRFReader(str(dstdir))

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
    return reader, d


if __name__ == "__main__":

    import sys
    import pprint
    dstdir = sys.argv[1]
    rdr, d = get_hardtarget_drf(dstdir)
    pprint.pprint(d)