import datetime
from pathlib import Path
import numpy as np
import digital_rf as drf
from hardtarget import drf_utils


####################################################################
# DIGITAL RF READERS
####################################################################


def is_pair_unpackable(a):
    """Returns true if a can be unpacked to a pair of variables."""
    return True if isinstance(a, (tuple, list)) and len(a) == 2 else False


# Cache reader instances by path
# TODO: this is a memory leak technically since it keeps a reference of the object here without deletion
_DRF_READERS = {}


def load_source(src):
    """
    Load Digital_rf reader object from (srcdir, chnl)
    """
    global _DRF_READERS
    if type(src) is not tuple:
        raise ValueError(f"tuple (path, chnl) expected, {src}")
    if not is_pair_unpackable(src):
        raise ValueError(f"tuple (path, chnl) expected, {src}")
    path, chnl = src
    if path is None or not Path(path).is_dir():
        raise ValueError(f"path must be directory, {path}")
    # read cached instance
    reader = _DRF_READERS.get(path, None)
    if reader is None:
        _DRF_READERS[path] = reader = drf.DigitalRFReader([str(path)])
    channels = reader.get_channels()
    if chnl not in channels:
        raise ValueError(f"reader does not support channel, {chnl}")
    return path, reader, chnl


####################################################################
# CALCULATE TASKS
####################################################################


def compute_job_tasks(job, n_tasks):
    """
    Generates a list of task indexes for given job.

    Parameters
    ----------
    job : dict
        job["N"] : int
            total number of jobs
        job["idx"] : int
            index of this job (idx < N)

    n_tasks: int
        total number of tasks

    Returns
    -------
    list
        List of task indexes

    Examples
    --------
    >>> get_tasks({"idx":1, "N:2"}, 8)
    [1,3,5,7]
    """
    return list(range(job["idx"], n_tasks, job["N"]))


def compute_total_tasks(ipp, n_ipp, num_cohints_per_file, bounds):
    """
    Compute the total number of tasks, associated with sample bounds.

    ipp: inter-pulse period length in samples
    n_ipp: number of interpulse periods to coherently integrate
    num_cohints_per_file:
      number of coherent integration periods to include in one output file
      smaller means that lower latency can be achieved
    """
    n_tasks = int(np.floor((bounds[1] - bounds[0]) / (ipp * n_ipp)) / num_cohints_per_file)
    return n_tasks


####################################################################
# BOUNDS
####################################################################


def compute_bounds(reader, chnl, sample_rate, start_time=None, end_time=None, relative_time=False):
    """
    optionally restrict sample bounds given timestamps
    """
    drf_bounds = reader.get_bounds(chnl)
    return drf_utils.time_interval_to_samples(
        start_time, end_time, drf_bounds, sample_rate, relative_time=relative_time
    )


####################################################################
# OUTPUT FILE PATH
####################################################################


def get_filepath(epoch_unix_us):
    """
    Generates a file path for h5 file to be written.

    Parameters
    ----------
    epoch_unix_us : int
        Epoch in unix time of file in microseconds

    Returns
    -------
    string
        filepath
    """
    dt = datetime.datetime.utcfromtimestamp(epoch_unix_us*1e-6)
    time_string = dt.strftime("%Y-%m-%dT%H-00-00")
    return Path(time_string) / f"gmf-{epoch_unix_us:08d}.h5"
