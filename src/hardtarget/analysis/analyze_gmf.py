import logging
import numpy as np
import time
import datetime
import digital_rf as drf
import h5py
from hardtarget.analysis import analyze_params
from hardtarget.analysis import analyze_ipps
from pathlib import Path


def get_tasks(job, n_tasks):
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


def bounds_to_str(bounds, sample_rate):
    """Create human readable representation of bounds, for logging."""
    _bounds = []
    for sample_number in bounds:
        dt = datetime.datetime.utcfromtimestamp(sample_number / sample_rate)
        _bounds.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
    return str(_bounds)


def pair_unpackable(a):
    """Returns true if a can be unpacked to a pair of variables."""
    return True if isinstance(a, (tuple, list)) and len(a) == 2 else False


####################################################################
# PRE-PROCESS
####################################################################


def preprocess(task):
    """
    Preprocess task.
    - checking parameters
    - generating derived parameters
    - compute a batch of processing subtasks

    Parameters
    ----------
    task: dict
        Dictionary containing a variety of parameters.

        output: string, optional, default None
            String path to directory for result files.
            If None, no files - return results as dictionary

    Returns
    -------
    progress: int
        progress in percent [0,100]

    result: dict
        dir: string
            path to directory with files
        files: list
            paths to each generated file
        out: dictionary with in-memory results
    """

    # logger
    logger = task.get("logger", logging.getLogger(__name__))

    # check and compute gmf params
    task["gmf_params"] = gmf_params = {
        **analyze_params.DEFAULT_PARAMS,
        **task.get("gmf_params", {}),
    }
    ok, msg = analyze_params.check_params(gmf_params)
    if not ok:
        logger.error(msg)
        return False
    analyze_params.process_params(gmf_params)

    # rx source
    rx = task.get("rx", None)
    if rx is None:
        logger.warning("missing rx")
        return False
    if not pair_unpackable(rx):
        logger.warning(f"rx must be tuple pair: {rx}")
        return False
    rx_path, rx_channel = rx
    if rx_path is None or not Path(rx_path).is_dir():
        logger.warning(f"rx src path does not exist: {rx_path}")
        return False

    rx_reader = drf.DigitalRFReader([rx_path])
    rx_channels = rx_reader.get_channels()
    if rx_channel not in rx_channels:
        logger.error(f"rdf data does not support rx_channel: {rx_channel}")
        return False
    task["rx"] = (rx_path, rx_channel)

    # tx source
    tx = task.get("tx", None)
    if tx is None:
        tx = rx
    if not pair_unpackable(tx):
        logger.warning(f"tx must be tuple pair: {tx}")
    tx_path, tx_channel = tx
    if tx_path is None:
        tx_path = rx_path
    if tx_channel is None:
        tx_channel = rx_channel
    if tx_path is None or not Path(tx_path).is_dir():
        logger.warning(f"tx src path does not exist: {tx_path}")
    tx_reader = drf.DigitalRFReader([tx_path])
    tx_channels = tx_reader.get_channels()
    if tx_channel not in tx_channels:
        logger.error(f"rdf data does not support tx_channel: {tx_channel}")
        return False
    task["tx"] = (tx_path, tx_channel)

    # output path
    output_path = task.get("output", None)
    if output_path is not None and not Path(output_path).is_dir():
        logger.warning(f"output folder does not exist: {output_path}")
        return False

    # check sample rate
    props = rx_reader.get_properties(rx_channel)
    # sample_rate = props["samples_per_second"].astype(np.int128)
    sample_rate = props["samples_per_second"].astype(np.int64)
    if sample_rate != gmf_params["sample_rate"]:
        logger.warning(f"rdf data only supports sample rate: {sample_rate}")
        return False
    gmf_params["sample_rate"] = sample_rate

    # check bounds rx channel
    bounds = list(rx_reader.get_bounds(rx_channel))
    start_time = gmf_params.get("start_time", None)
    end_time = gmf_params.get("end_time", None)
    if start_time is not None:
        _b0 = int(start_time * sample_rate)
        print(_b0, bounds[0])
        assert _b0 >= bounds[0], "Given start time is before input data start"
        bounds[0] = _b0
    if end_time is not None:
        _b1 = int(end_time * sample_rate)
        assert _b1 <= bounds[1], "Given end time is after input data end"
        bounds[1] = _b1
    gmf_params["bounds"] = bounds = tuple(bounds)

    # check blocks rx channel
    blocks = rx_reader.get_continuous_blocks(bounds[0], bounds[1], rx_channel)
    if len(blocks) > 1:
        logger.warning(f"multiple continuous blocks: {len(blocks)}")

    # compute tasks
    # inter-pulse period length in samples
    ipp = gmf_params["ipp"]
    # number of interpulse periods to coherently integrate
    n_ipp = gmf_params["n_ipp"]
    # number of coherent integration periods to include in one output file
    # smaller means that lower latency can be achieved
    num_cohints_per_file = gmf_params["num_cohints_per_file"]
    n_tasks = int(np.floor((bounds[1] - bounds[0]) / (ipp * n_ipp)) / num_cohints_per_file)

    # compute subset of tasks for this job
    job = task.get("job", {"idx": 0, "N": 1})
    task["job_tasks"] = job_tasks = get_tasks(job, n_tasks)

    # logging
    logger.debug(f"channel: {rx_channel}")
    logger.debug(f"bounds: {bounds_to_str(bounds, sample_rate)}")
    logger.debug(f"sample rate: {sample_rate}")
    logger.debug(f"continuous blocks: {len(blocks)}")
    logger.info(f"total_tasks: {n_tasks}")
    logger.debug(f"job_tasks: {len(job_tasks)}")

    return True


####################################################################
# PROCESS
####################################################################


def process(task):
    """
    Analyze data using gmf.

    Parameters
    ----------
    task: dict
        Dictionary containing a variety of parameters.

        output: string, optional, default None
            String path to directory for result files.
            If None, no files - return results as dictionary

    Returns
    -------
    progress: int
        progress in percent [0,100]

    result: dict
        dir: string
            path to directory with files
        files: list
            paths to each generated file
        out: dictionary with in-memory results
    """
    logger = task.get("logger", logging.getLogger(__name__))
    progress = task.get("progress", None)
    clobber = task.get("clobber", False)

    rx = task["rx"]
    tx = task["tx"]
    output_path = task.get("output", None)
    gmf_params = task["gmf_params"]

    # number of range-gates to analyze
    n_range_gates = gmf_params["n_range_gates"]
    # inter-pulse period length in samples
    ipp = gmf_params["ipp"]
    # number of interpulse periods to coherently integrate
    n_ipp = gmf_params["n_ipp"]
    # number of coherent integration periods to include in one output file
    # smaller means that lower latency can be achieved
    num_cohints_per_file = gmf_params["num_cohints_per_file"]
    # bounds
    bounds = gmf_params["bounds"]
    # sample rate
    sample_rate = gmf_params["sample_rate"]
    # subtasks
    job_tasks = task["job_tasks"]
    n_job_tasks = len(job_tasks)

    # logging
    job = task.get("job", {"idx": 1, "N": 1})
    logger.info(f"starting job {job['idx']}/{job['N']} with {n_job_tasks} tasks")

    # process
    results = {"dir": output_path, "files": [], "out": {}}
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

        # write to file if output_path is defined
        if output_path is not None:
            outfile = Path(output_path) / filepath
            # crate directory
            dirname = Path(outfile).parent
            dirname.mkdir(parents=True, exist_ok=True)

            if outfile.is_file() and not clobber:
                logger.info(f"job already done {job['idx']}/{job['N']}")
                return 100, results

            # write result
            out = h5py.File(outfile, "w")
        else:
            out = {}

        # process
        ts0 = time.time()
        for i in range(num_cohints_per_file):
            i0 = file_idx + i * ipp * n_ipp
            result = analyze_ipps.analyze_ipps(rx, tx, i0, gmf_params)
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

        # output
        out["gmf"] = gmf_max
        out["gmf_dc"] = gmf_dc
        out["r"] = gmf_params["ranges"].copy()
        out["a"] = gmf_a
        out["v"] = gmf_v
        out["tx_pwr"] = gmf_txp
        out["i0"] = i0

        if output_path is not None:
            out.close()
            results["files"].append(filepath.name)
        else:
            results["out"][filepath.name] = out

        # progress
        if progress is not None:
            progress(idx + 1, n_job_tasks)

    logger.info(f"finishing job {job['idx']}/{job['N']} with {n_job_tasks} tasks")
    return 100, results
