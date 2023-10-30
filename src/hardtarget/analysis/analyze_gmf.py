import logging
import numpy as np
import time
import datetime
import digital_rf as drf
import h5py
from tqdm import tqdm
from pathlib import Path
from hardtarget.analysis import analyze_ipps
from hardtarget import gmf_utils, drf_utils


####################################################################
# DIGITAL RF READERS
####################################################################

def is_pair_unpackable(a):
    """Returns true if a can be unpacked to a pair of variables."""
    return True if isinstance(a, (tuple, list)) and len(a) == 2 else False


# Cache reader instances by path
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
        _DRF_READERS[path] = reader = drf.DigitalRFReader([path])
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


def compute_total_tasks(gmf_params, bounds):
    """
    Compute the total number of tasks, associated with sample bounds.
    """
    # inter-pulse period length in samples
    ipp = gmf_params["ipp"]
    # number of interpulse periods to coherently integrate
    n_ipp = gmf_params["n_ipp"]
    # number of coherent integration periods to include in one output file
    # smaller means that lower latency can be achieved
    num_cohints_per_file = gmf_params["num_cohints_per_file"]
    n_tasks = int(np.floor((bounds[1] - bounds[0]) / (ipp * n_ipp)) / num_cohints_per_file)
    return n_tasks


####################################################################
# BOUNDS
####################################################################

def compute_bounds(reader, chnl, sample_rate,
                   start_time=None,
                   end_time=None,
                   relative_time=False):
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

def get_filepath(file_idx, sample_rate):
    """
    Generates a file path for h5 file to be written.

    Parameters
    ----------
    file_idx : int
        index associate with output file
    sample_rate: int
        sample rate for processed data

    Returns
    -------
    string
        filepath
    """
    dt = datetime.datetime.utcfromtimestamp(file_idx / sample_rate)
    time_string = dt.strftime("%Y-%m-%dT%H-00-00")
    return Path(time_string) / f"gmf-{file_idx:08d}.h5"


####################################################################
# ANALYZE GMF
####################################################################


def analyze_gmf(rx,
                tx,
                config=None,
                start_time=None,
                end_time=None,
                relative_time=False,
                job={"idx": 1, "N": 1},
                logger=None,
                progress=False,
                progress_position = 0,
                clobber = False,
                output=None,
                gmflib=None
                ):
    """
    Analyze data using gmf.

    Parameters
    ----------
    rx: Digital_rf reader
    tx: Digital_rf reader
    gmf_config: path gfm config file
    job: used to identify subset of task indexes for this job
 
    Returns
    -------

    result: dict
        dir: string
            path to directory with files
        files: list
            paths to each generated file
        out: dictionary with in-memory results
    """
    
    # load data sources
    rx_srcdir, rx_reader, rx_chnl = load_source(rx)
    tx_srcdir, tx_reader, tx_chnl = load_source(tx)

    # gmf params
    gmf_params = gmf_utils.load_gmf_params(rx_srcdir, config)
    # override config file
    if gmflib is not None:
        gmf_params['gmflib'] = gmflib
    logger.info("Using GMF backend: " + gmf_params["gmflib"])

    # bounds
    bounds = compute_bounds(rx_reader, rx_chnl, gmf_params["sample_rate"],
                            start_time=start_time,
                            end_time=end_time,
                            relative_time=relative_time)

    # tasks
    total_tasks = compute_total_tasks(gmf_params, bounds)
    job_tasks = compute_job_tasks(job, total_tasks)
    tasks_skipped = 0

    # logging
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"starting job {job['idx']}/{job['N']} with {len(job_tasks)} tasks")

    # progress
    if progress:
        progress_bar = tqdm(
            position=progress_position,
            desc="Processing",
            total=len(job_tasks),
        )

    # variable access
    n_range_gates = gmf_params["n_range_gates"]
    ipp = gmf_params["ipp"]
    n_ipp = gmf_params["n_ipp"]
    num_cohints_per_file = gmf_params["num_cohints_per_file"]
    sample_rate = gmf_params["sample_rate"]

    # process
    results = {"dir": output, "files": [], "out": {}}
    for idx, task_idx in enumerate(job_tasks):
        # initialise
        gmf_max = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_dc = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_v = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_a = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_txp = np.zeros(num_cohints_per_file, dtype=np.float32)

        # filename outfile
        file_idx = task_idx * ipp * n_ipp * num_cohints_per_file + bounds[0]
        filepath = get_filepath(file_idx, sample_rate)

        # write to file if output is defined
        if output is not None:
            outfile = Path(output) / filepath
            # crate directory
            dirname = Path(outfile).parent
            dirname.mkdir(parents=True, exist_ok=True)
            if outfile.is_file() and not clobber:
                logger.debug(f"job {job['idx']}/{job['N']} already done task {idx}/{len(job_tasks)}")
                tasks_skipped += 1
                continue

        # process
        ts0 = time.time()
        for i in range(num_cohints_per_file):
            i0 = file_idx + i * ipp * n_ipp
            result = analyze_ipps.analyze_ipps((rx_reader, rx_chnl), 
                                               (tx_reader, tx_chnl), 
                                               i0, gmf_params)
            gmf_max[i, :], gmf_dc[i, :], gmf_v[i, :], gmf_a[i, :], gmf_txp[i] = result
            # rgi = np.argmax(gmf_max[i, :])
        ts1 = time.time()

        # log
        info = {
            "task": task_idx,
            "time": ts1 - ts0,
            "real": (ts1 - ts0) / (n_ipp * ipp / sample_rate),
        }
        msg = "task_idx {task:4} time {time:1.2f} cpu/real {real:1.2f}"
        logger.debug(msg.format(**info))

        if output is not None:
            # write result
            out = h5py.File(outfile, "w")
        else:
            out = {}

        # output
        out["gmf"] = gmf_max
        out["gmf_dc"] = gmf_dc
        out["r"] = gmf_params["ranges"].copy()
        out["a"] = gmf_a
        out["v"] = gmf_v
        out["tx_pwr"] = gmf_txp
        out["i0"] = i0

        if output is not None:
            out.close()
            results["files"].append(filepath.name)
        else:
            results["out"][filepath.name] = out

        # progress
        if progress:
            progress_bar.update(1 + tasks_skipped)
            tasks_skipped = 0
    if progress:
        progress_bar.close()

    logger.info(f"finishing job {job['idx']}/{job['N']} with {len(job_tasks)} tasks")
    return results
