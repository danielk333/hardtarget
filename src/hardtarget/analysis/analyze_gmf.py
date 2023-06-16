import logging
import numpy as np
import os
import time
import datetime
import digital_rf as drf
import h5py
from hardtarget.analysis import analyze_params
from hardtarget.analysis import analyze_ipps

MODULO_PROGRESS = 1


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
    return os.path.join(*[time_string, f"gmf-{file_idx:08d}.h5"])


####################################################################
# ANALYSE GMF
####################################################################


def process(task, clobber=False):
    """
    Analyze data using gmf.

    Parameters
    ----------
    task: dict
        Dictionary containing a variety of parameters.
    clobber: bool, optional
        If True, overwrite pre-existing files (defalt False)

    Returns
    -------
    progress: int
        progress in percent [0,100]

    result: dict
        dir: string
            path to directory with files
        files: list
            paths to each generated file
    """

    job = task.get("job", {"idx": 0, "N": 1})
    logger = task.get("logger", logging.getLogger(__name__))

    # path to directory with drf data
    input_path = task.get("input", None)
    # path to directory for writing output
    output_path = task.get("output", None)

    # process
    results = {"dir": output_path, "files": []}

    # check paths
    if input_path is None or not os.path.isdir(input_path):
        logger.warning(f"input folder does not exist: {input_path}")
        return 0, results
    if output_path is None or not os.path.isdir(output_path):
        logger.warning(f"output folder does not exist: {output_path}")
        return 0, results

    # gmf params
    gmf_params = {**analyze_params.DEFAULT_PARAMS, **task.get("gmf_params", {})}
    # computing derived parameters
    ok, msg = analyze_params.check_params(gmf_params)
    if not ok:
        logger.error(msg)
        return 0, results

    analyze_params.process_params(gmf_params)

    logger.info(f"starting job {job['idx']}/{job['N']}")

    # read drf data
    rdf_reader = drf.DigitalRFReader([input_path])

    # inter-pulse period length in samples
    ipp = gmf_params["ipp"]
    # number of interpulse periods to coherently integrate
    n_ipp = gmf_params["n_ipp"]
    # number of coherent integration periods to include in one output file
    # smaller means that lower latency can be achieved
    num_cohints_per_file = gmf_params["num_cohints_per_file"]
    # number of range-gates to analyze
    n_range_gates = gmf_params["n_range_gates"]
    # optional bounds
    start_time = gmf_params.get("start_time", None)
    end_time = gmf_params.get("end_time", None)

    # channel
    tx_channel = gmf_params["tx_channel"]
    rx_channel = gmf_params["rx_channel"]
    chnls = rdf_reader.get_channels()
    if tx_channel not in chnls:
        logger.error(f"rdf data does not support tx_channel: {tx_channel}")
        return 0, results
    if rx_channel not in chnls:
        logger.error(f"rdf data does not support rx_channel: {rx_channel}")
        return 0, results
    chnl = rx_channel

    # sample rate
    props = rdf_reader.get_properties(chnl)
    sample_rate = props["samples_per_second"].astype(np.int128)
    if sample_rate != gmf_params["sample_rate"]:
        logger.warning(f"rdf data only supports sample rate: {sample_rate}")
        return 0, results

    # bounds
    bounds = list(rdf_reader.get_bounds(chnl))
    # adjust lower bound
    if start_time is not None:
        _b0 = int(start_time * sample_rate)
        assert _b0 >= bounds[0], "Given start time is before input data start"
        bounds[0] = _b0
    if end_time is not None:
        _b1 = int(end_time * sample_rate)
        assert _b1 <= bounds[1], "Given end time is after input data end"
        bounds[1] = _b1

    bounds = tuple(bounds)

    # blocks
    blocks = rdf_reader.get_continuous_blocks(bounds[0], bounds[1], chnl)
    if len(blocks) > 1:
        logger.warning(f"multiple continuous blocks: {len(blocks)}")

    # tasks
    n_tasks = int(np.floor((bounds[1] - bounds[0]) / (ipp * n_ipp)) / num_cohints_per_file)

    # subset of tasks for this job
    job_tasks = get_tasks(job, n_tasks)
    n_job_tasks = len(job_tasks)

    # logging

    def to_str(value):
        dt = datetime.datetime.utcfromtimestamp(value / sample_rate)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    bounds_str = str([to_str(e) for e in bounds])

    logger.debug(f"channel: {chnl}")
    logger.debug(f"bounds: {bounds_str}")
    logger.debug(f"sample rate: {sample_rate}")
    logger.debug(f"continuous blocks: {len(blocks)}")
    logger.info(f"total_tasks: {n_tasks}")
    logger.debug(f"job_tasks: {n_job_tasks}")

    for idx, task_idx in enumerate(job_tasks):
        # progress
        if idx == n_job_tasks - 1 or idx % MODULO_PROGRESS == 0:
            logger.info(f"write progress {idx}/{n_job_tasks}")

        # initialise
        gmf_max = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_dc = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_v = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_a = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_txp = np.zeros(num_cohints_per_file, dtype=np.float32)

        # filename outfile
        file_idx = task_idx * ipp * n_ipp * num_cohints_per_file + bounds[0]
        filepath = get_filepath(file_idx, sample_rate)
        results["files"].append(filepath)
        outfile = os.path.join(output_path, filepath)
        # crate directory
        dirname, _ = os.path.split(outfile)
        os.makedirs(dirname, exist_ok=True)

        if not os.path.isfile(outfile) or clobber:
            ts0 = time.time()

            # process
            for i in range(num_cohints_per_file):
                i0 = file_idx + i * ipp * n_ipp
                result = analyze_ipps.analyze_ipps(rdf_reader, i0, gmf_params)
                gmf_max[i, :], gmf_dc[i, :], gmf_v[i, :], gmf_a[i, :], gmf_txp[i] = result
                # rgi = np.argmax(gmf_max[i, :])

            # log
            ts1 = time.time()
            info = {
                "task": task_idx,
                "time": ts1 - ts0,
                "real": (ts1 - ts0) / (n_ipp * ipp / sample_rate),
            }
            msg = "task_idx {task:4} time {time:1.2f} cpu/real {real:1.2f}"
            logger.info(msg.format(**info))

            # write result
            out = h5py.File(outfile, "w")
            out["gmf"] = gmf_max
            out["gmf_dc"] = gmf_dc
            out["r"] = gmf_params["ranges"].copy()
            out["a"] = gmf_a
            out["v"] = gmf_v
            out["tx_pwr"] = gmf_txp
            out["i0"] = i0
            out.close()

    logger.info(f"finishing job {job['idx']}/{job['N']}")
    return 100, results
