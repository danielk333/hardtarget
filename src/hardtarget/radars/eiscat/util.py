import bz2
import re
from pathlib import Path
import datetime as dt
import numpy as np
import scipy.io as sio
from hardtarget.radars.eiscat import load_expconfig
from hardtarget.utils import str_from_ts

####################################################################
# GLOBALS
####################################################################

PARBL_ELEVATION = 8
PARBL_AZIMUTH = 9
PARBL_END_TIME = 10
PARBL_SEQUENCE = 11
PARBL_START_TIME = 42  # upar[1]
PARBL_RADAR_FREQUENCY = 54  # upar[13]


####################################################################
# UTIL
####################################################################

def index_of(alist, match_func):
    """
    Returns index of first object in list which matches given match func
    """
    matches = [match_func(e) for e in alist]
    return matches.index(True)


def to_i2x16(zz):
    """
    convert from (N,) complex128 to (N,2) int16
    """
    zz2x16 = np.empty((len(zz), 2), dtype=np.int16)
    zz2x16[:, 0] = zz.real.astype(np.int16)
    zz2x16[:, 1] = zz.imag.astype(np.int16)
    return zz2x16


def loadmat(path):
    """
    Loat matlab file.
    Transparently unzips files on the fly (not in the filesystem)
    """
    if path.endswith(".mat.bz2"):
        path = bz2.open(path, "rb")
    return sio.loadmat(path)


def expinfo_split(xpinf):
    """
    Move from hard coded constants to loading config based on exp name/version
    'kst0 leo_bpark_2.1u_NO' -> ('kst0', 'leo_bpark', '2.1u', 'NO')
    """
    try:
        # host, name, versi, owner = \
        match = re.match(r"(\w+) +(\w+)_(\d+(?:\.\d+)?[a-z]*)_(\w+)", xpinf)
        return match.groups()
    except Exception as e:
        raise ValueError(f"d_ExpInfo: {xpinf} not understood: {e}")


def parse_foldername(product_folder):
    """
    Parses foldername of eiscat raw data product to produce (expname, expvers, chnl)
        e.g. foldername "leo_pwait_2.3r_3P@sod"
        e.g. result ('leo_pwait', '2.3r', 'sod')

    NOTE - this is similar to expinfo split - which produces similar output
    from parsing eiscat metadata. This function additionally
    gets the chnl name. This function would be obsolete, if correct chnl
    name was parsed from metadata instead. At this time, we are useing the
    folder as a quick solution.
    """
    product_folder = Path(product_folder)
    tokens = product_folder.name.split("_")
    expname = "_".join(tokens[:2])
    expvers = tokens[2]
    chnl = tokens[3].split("@")[1]
    return expname, expvers, chnl


def beginning_of_year(_dt):
    """Given a datetime, returns a new datetime representing the beginning of the year"""
    return _dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def get_seconds_since_year_start(_dt):
    """Given a datetime, returns the number of seconds into the year"""
    _dt_year_start = beginning_of_year(_dt)
    return int((_dt - _dt_year_start).total_seconds())


def get_filedate_from_sample_file_end(sample_file_end, sample_rate):
    # filename date is calculated from file_end (first sample of next file)
    ts = sample_file_end / sample_rate
    _dt = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    return get_seconds_since_year_start(_dt)


def eiscat_file_idx_by_datetime(files, _dt):
    """
    Return index of eiscat file covering ts
    """

    # check that _dt is same year as files
    year = int(files[0].parent.name[:4])
    if year != _dt.year:
        raise Exception(f"mismatch year: dt year {_dt.year}, files year {year}")

    # find index of first file which offset > offset of _dt
    offset = get_seconds_since_year_start(_dt)

    def match_start(file):
        _offset = int(file.name.split(".")[0])
        return _offset > offset

    return index_of(files, match_start)


####################################################################
# EISCAT FILES
####################################################################

def eiscat_files(productpath, start=None, end=None, count=None):
    """
    return sorted list of eiscat files from product
    file list is limited by start && (end || count)
    start and end may be indexes
    - (int) index into product file list
    - (str) filename_date (seconds since year start)
    - (datetime) utc time
    count (int) specifies number of files to return (from start) and overrides end
    """
    def ok(f):
        return f.is_file and f.name.endswith(".mat.bz2")

    subdirs = [d for d in productpath.iterdir() if d.is_dir()]
    files = []
    for subdir in subdirs:
        files += [f for f in subdir.iterdir() if ok(f)]

    # sort by path (ascending in time)
    files.sort()

    # start
    if isinstance(start, str):
        # start is str date (year offset in seconds) from filename
        def match(f):
            return start == f.name.split(".")[0]
        start_idx = index_of(files, match)
    elif isinstance(start, dt.datetime):
        start_idx = eiscat_file_idx_by_datetime(files, start)
    elif start is None:
        start_idx = 0
    elif isinstance(start, int):
        # start is index
        start_idx = start
    else:
        raise Exception("illegal start", start)

    if end is None and count is None:
        # only start given
        return files[start_idx, start_idx+1]

    # end
    if end is not None:
        if isinstance(end, str):
            # end is str date from filename
            def match(f):
                return end == f.name.split(".")[0]
            end_idx = index_of(files, match) + 1
        elif isinstance(end, dt.datetime):
            end_idx = eiscat_file_idx_by_datetime(files, end)
        elif end is None:
            end_idx = len(files)
        elif isinstance(end, int):
            # end is index
            end_idx = end
        else:
            raise Exception("illegal end", end)

    # count has precedence over end
    if count is not None:
        end_idx = min(start_idx + count, len(files))

    return files[start_idx:end_idx]


####################################################################
# EISCAT LOAD
####################################################################

def eiscat_load_file(filepath):
    """
    Returns a dictionary with meta information for given eiscat file
    """
    mat = loadmat(str(filepath))
    host, expname, expvers, owner = expinfo_split(str(mat["d_ExpInfo"][0]))
    cfg = load_expconfig(expname)
    cfv = cfg[expvers]
    sample_rate = float(cfv.get("sample_rate"))
    file_secs = float(cfv.get("file_secs"))
    samples_per_file = int(file_secs * sample_rate)
    chnl = parse_foldername(filepath.parent.parent)[2]

    # global start time for sampling (repeated for all files)
    ts_origin_sec = float(mat["d_parbl"][0][PARBL_START_TIME])
    # sample index corresponding to global start time
    idx_origin = int(np.floor(ts_origin_sec * sample_rate))
    # end time of file
    ts_endfile_sec = float(mat["d_parbl"][0][PARBL_END_TIME])

    # NOTE: ts_endfile_sec can not be trusted to be precisely
    # consistent with ts_origin_sec - as a source for sample number.
    # we use the approach of counting samples from ts_orgin_sec,
    # relying on a fixed sample count per file

    # NOTE: file_idx is a logical sequence number for files, starting from
    # file_idx 0 at ts_origin_sec. This does not correspond to the order of files processed.
    # For instance, the first file might have file_idx 6, indicating that the
    # recording only started some time after ts_origin_sec
    # also the inprecision in ts_endfile_sec goes away in division
    file_idx = round((ts_endfile_sec - ts_origin_sec) / file_secs) - 1
    # sample index for start of first file
    samples_per_file = int(file_secs * sample_rate)
    # index of filestart
    idx_startfile = idx_origin + file_idx * samples_per_file

    # check that samples is correct
    data = to_i2x16(mat["d_raw"][:, 0])
    n_samples = len(data)

    errors = []

    # check sample count in data
    if n_samples != samples_per_file:
        errors.append(f'incorrect sample number {n_samples}')

    # file not aligned with logical file boundaries
    remainder = (idx_startfile - idx_origin) % samples_per_file
    if remainder != 0.0:
        errors.append(f"misalignment with index space {remainder}")

    # check if filename is consistent with read index
    # filename corresponds to end_index (first index of next file)
    offset = get_filedate_from_sample_file_end(idx_startfile + samples_per_file, sample_rate)
    _offset = int(filepath.name.split(".")[0])

    if offset != _offset:
        errors.append(f"filename inconsistency {offset} {_offset}")

    meta = {
        "exp": {
            "name": expname,
            "version": expvers,
            "chnl": chnl,
            "sample_rate": sample_rate,
            "samples_per_file": n_samples,
            "file_secs": file_secs,
            "radar_frequency": float(mat["d_parbl"][0][PARBL_RADAR_FREQUENCY]),
            "ipp": cfv.get("ipp"),
            "tx_pulse_length": cfv.get("tx_pulse_length"),
            "rx_start": cfv.get("rx_start"),
            "rx_end": cfv.get("rx_end"),
            "tx_start": cfv.get("tx_start"),
            "tx_end": cfv.get("tx_end"),
            "cal_on": cfv.get("cal_on"),
            "cal_off": cfv.get("cal_off")
        },
        "ts": {
            "origin": ts_origin_sec,
            "file_start": ts_endfile_sec - file_secs,
            "file_end": ts_endfile_sec,
        },
        "date": {
            "origin": str_from_ts(ts_origin_sec),
            "file_start": str_from_ts(ts_endfile_sec - file_secs),
            "file_end": str_from_ts(ts_endfile_sec),
        },
        "sample": {
            "origin": idx_origin,
            "file_start": idx_startfile,
            "file_end": (idx_startfile + samples_per_file)
        },
        "file": {
            "name": filepath.name,
            "idx": file_idx
        },
        "errors": errors
    }

    pointing_data = {
        'azimuth': float(mat["d_parbl"][0][PARBL_AZIMUTH]) % 360,
        'elevation': float(mat["d_parbl"][0][PARBL_ELEVATION])
    }

    return meta, data, pointing_data


####################################################################
# EISCAT PROCESS
####################################################################

def default_pad(n_pad, file):
    print(f"pad data {n_pad} samples for {file.name}")


def default_write(data, file):
    print(f'write data {len(data)} samples for {file.name}')


def default_drop(data, file):
    print(f"drop data {len(data)} samples for {file.name}")


def default_pointing(sample, pointing_data):
    azimuth = pointing_data["azimuth"]
    elevation = pointing_data["elevation"]
    print(f"write pointing {azimuth} {elevation}")


def default_progress(idx, n_files, period=10):
    if idx + 1 == n_files or idx % period == 0:
        print(f"write progress {idx+1}/{n_files}")


def eiscat_process(files,
                   write_data=None,
                   pad_data=None,
                   drop_data=None,
                   write_pointing=None,
                   log_progress=None):
    """
    Process sequence of eiscat files
    """
    if write_data is None:
        write_data = default_write
    if pad_data is None:
        pad_data = default_pad
    if drop_data is None:
        drop_data = default_drop

    # sample idx of next write
    sample_write = None

    for idx, file in enumerate(files):

        if log_progress:
            log_progress(idx, len(files), period=10)

        meta, data, pointing_data = eiscat_load_file(file)

        sample_file_start = meta["sample"]["file_start"]
        errors = meta['errors']

        if sample_write is None:
            sample_write = sample_file_start

        if not errors:

            # check that we are not writing old data
            if sample_file_start < sample_write:
                errors.append("attempt to overwrite data")
            # check if zero padding is needed
            n_pad = sample_file_start - sample_write
            if n_pad > 0:
                pad_data(n_pad, file)

            # write chunk to stream
            write_data(data, file)
            # write pointint data
            if write_pointing:
                write_pointing(sample_write, pointing_data)

        else:
            drop_data(data, file)
            yield meta
            # do not increment sample_write
            continue

        # increment sample_write
        sample_write = sample_file_start + meta["exp"]["samples_per_file"]

        yield meta
