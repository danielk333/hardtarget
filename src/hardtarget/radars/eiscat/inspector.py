
from pathlib import Path
import datetime as dt
import numpy as np
from hardtarget.radars.eiscat.convert import loadmat, parse_foldername
from hardtarget.radars.eiscat.convert import load_expconfig, expinfo_split
from hardtarget.radars.eiscat.convert import to_i2x16
from hardtarget.radars.eiscat.convert import get_seconds_since_year_start
from hardtarget.radars.eiscat.convert import PARBL_START_TIME, PARBL_END_TIME

def get_date_str(ts_sec):
    _datetime = dt.datetime.fromtimestamp(ts_sec, tz=dt.timezone.utc)
    return _datetime.strftime('%Y-%m-%dT%H:%M:%S')


def eiscat_inspect_file (idx, filepath):
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
    zz = to_i2x16(mat["d_raw"][:, 0])
    n_samples = len(zz)

    errors = []

    # check sample count in data
    if n_samples != samples_per_file:
        errors.append("incorrect sample number")

    # file not aligned with logical file boundaries
    remainder = (idx_startfile - idx_origin) % samples_per_file
    if remainder != 0.0:
        errors.append("misalignment with index space")

    # check if filename is consistent with read index
    # filename corresponds to end_index (first index of next chunk)
    idx_next = idx_startfile + samples_per_file
    ts_next = idx_next / 1e6
    dt_next = dt.datetime.fromtimestamp(ts_next, tz=dt.timezone.utc)
    offset = get_seconds_since_year_start(dt_next)
    _offset = int(filepath.name.split(".")[0])
    if offset != _offset:
        errors.append("filename inconsistency")

    return {
        "name": filepath.name,
        "idx": idx,
        "expname": expname,
        "expvers": expvers,
        "chnl": chnl,
        "sample_rate": sample_rate,
        "file_secs": file_secs,
        "samples_per_file": samples_per_file,
        "ts": {
            "origin_sec": ts_origin_sec,
            "startfile_sec": ts_endfile_sec - file_secs,
            "startfile_date": get_date_str(ts_endfile_sec - file_secs),
            "endfile_sec": ts_endfile_sec,
            "endfile_date": get_date_str(ts_endfile_sec),
        },
        "samples": {
            "origin": idx_origin,
            "file": idx_startfile,
            "next": (idx_startfile + samples_per_file)
        },
        "file_idx": file_idx,
        "errors": errors
    }, zz



def eiscat_inspect (productpath, start=None, end=None, count=None):
    """
    info from eiscat raw data
    """ 
    files = eiscat_files(productpath, start=start, end=end, count=count)
    for idx, file in files:
        yield eiscat_inspect_file(idx, file)


def index_of (alist, match_func):
    matches = [match_func(e) for e in alist]
    return matches.index(True)


def eiscat_files (productpath, start=None, end=None, count=None):
    """
    return sorted list of eiscat files from product
    file list is limited by start && (end || count)
    start and end may be indexes (int) or data (str)
    """
    def ok(f):
        return f.is_file and f.name.endswith(".mat.bz2")

    subdirs = [d for d in productpath.iterdir() if d.is_dir()]
    files = []
    for subdir in subdirs:
        files +=  [f for f in subdir.iterdir() if ok(f)]

    # sort by path (ascending in time)
    files.sort()

    # start
    if isinstance(start, str):
        # start is datestr
        def match(f):
            return start == f.name.split(".")[0]
        start = index_of(files, match)
    elif start is None:
        start = 0
    elif isinstance(start, int):
        pass
    else:
        raise Exception("illegal start", start)

    # end
    if isinstance(end, str):
        # start is datestr
        def match(f):
            return end == f.name.split(".")[0]
        end = index_of(files, match) + 1
    elif end is None:
        end = len(files)
    elif isinstance(end, int):
        pass
    else:
        raise Exception("illegal end", end)

    if count:
        end = min(start + count, len(files))

    files = list(enumerate(files))
    return files[start:end]



if __name__ == "__main__":


    # test 
    PROJECT = Path("/cluster/work/users/inar/mode-test")
    RAW = PROJECT / "raw"
    DRF = PROJECT / "drf"
    SOURCE = RAW / "leo_bpark_2.2_EI-20151027-42m/leo_bpark_2.2_EI@42m"
    
    import pprint
    for info, zz in eiscat_inspect(SOURCE, start=193, count=2):
        pprint.pprint(info, sort_dicts=False)
