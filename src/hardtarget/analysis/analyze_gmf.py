import logging
import numpy as np
import os
import datetime
import digital_rf as drf
from hardtarget.analysis import analyze_params
from hardtarget.analysis import analyze_ipps


def get_tasks (job, n_tasks):
    """
    returns a list of task indexes for given job

    n_tasks: (int) total number of tasks

    job["N"] : total number of jobs
    job["idx"] : index of this job (idx < N)

    compute tasks for given job
    """
    return list(range(job["idx"], n_tasks, job["N"]))



def get_filepath(dir, file_idx, sample_rate):
    """
    create a file path for output
    """
    dt = datetime.datetime.utcfromtimestamp(file_idx/sample_rate)
    time_string = dt.strftime("%Y-%m-%dT%H-00-00")
    return os.path.join(*[
        dir,
        time_string,
        f"gmf-{file_idx:08d}.h5"
        ])


####################################################################
# ANALYSE GMF
####################################################################

def process(task):
    """
    Analyze data using gmf.
    """
    
    job = task.get("job", {"idx": 0, "N": 1})
    logger = task.get("logger", logging.getLogger(__name__))
    
    # path to directory with drf data
    input = task.get("input", None)
    # path to directory for writing output
    output = task.get("output", None)

    # check paths
    if input is None or not os.path.isdir(input):
        logger.warning(f"input folder does not exist: {input}")
        return 0, {}
    if output is None or not os.path.isdir(output):
        logger.warning(f"output folder does not exist: {output}")
        return 0, {}

    # gmf params
    gmf_params = {**analyze_params.DEFAULT_PARAMS, **task.get("gmf_params", {})}
    # computing derived parameters
    analyze_params.process_params(gmf_params)
    ok, msg = analyze_params.check_params(gmf_params)
    if not ok:
        logger.error(msg)
        return 0, {}

    logger.info(f"starting job {job['idx']}/{job['N']}")

    # read drf data
    rd = drf.DigitalRFReader([input])

    # channel
    tx_channel = gmf_params["tx_channel"]
    rx_channel = gmf_params["rx_channel"]
    chnls = rd.get_channels()
    if not tx_channel in chnls:
        logger.error(f"rdf data does not support tx_channel: {tx_channel}")
        return 0, {}
    if not rx_channel in chnls:
        logger.error(f"rdf data does not support rx_channel: {rx_channel}")
        return 0, {}
    chnl = rx_channel

    # bounds
    bounds = rd.get_bounds(chnl)

    # sample rate
    props = rd.get_properties(chnl)
    sample_rate = props["samples_per_second"].astype(int)
    if sample_rate != gmf_params["sample_rate"]:
        logger.warning(f"rdf data only supports sample rate: {sample_rate}")
        return 0, {}

    # blocks
    blocks = rd.get_continuous_blocks(bounds[0], bounds[1], chnl)
    if len(blocks) > 1:
        logger.warning(f"multiple continuous blocks: {len(blocks)}")

    # inter-pulse period length in samples
    ipp = gmf_params["ipp"]
    # number of interpulse periods to coherently integrate
    n_ipp = gmf_params["n_ipp"]
    # number of coherent integration periods to include in one output file
    # smaller means that lower latency can be achieved
    num_cohints_per_file = gmf_params["num_cohints_per_file"]
    # number of range-gates to analyze
    n_range_gates = gmf_params["n_range_gates"]
    # optional lower bound
    t0 = gmf_params.get("t0", None)

    
    # adjust lower bound
    if t0 != None:
        bounds[0] = int(t0*sample_rate)
        # TODO sanity check bounds

    # tasks
    n_tasks = int(np.floor((bounds[1]-bounds[0])/(ipp*n_ipp))/num_cohints_per_file)

    # subset of tasks for this job
    job_tasks = get_tasks(job, n_tasks)[:40]
    n_job_tasks = len(job_tasks)

    # logging 

    def to_str(value):
        dt = datetime.datetime.utcfromtimestamp(value/sample_rate)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    bounds_str = str([to_str(e) for e in bounds])

    logger.debug(f"channel: {chnl}")
    logger.debug(f"bounds: {bounds_str}")
    logger.debug(f"sample rate: {sample_rate}")
    logger.debug(f"continuous blocks: {len(blocks)}")
    logger.debug(f"total_tasks: {n_tasks}")
    logger.debug(f"job_tasks: {n_job_tasks}")

    # initialise 
    gmf_max = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    gmf_dc = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    gmf_v = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    gmf_a = np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    gmf_txp = np.zeros(num_cohints_per_file, dtype=np.float32)

    # process
    for idx, task_idx in enumerate(job_tasks):
        # progress
        if idx == n_job_tasks-1 or idx % 10 == 0:
            logger.info(f"write progress {idx}/{n_job_tasks}")
        file_idx = task_idx*ipp*n_ipp*num_cohints_per_file + bounds[0]        
        outfile = get_filepath(output, file_idx, sample_rate)        
        if not os.path.isfile(outfile):
            for i in range(num_cohints_per_file):
                i0 = file_idx + i*ipp*n_ipp
                # result = analyze_ipps.analyze_ipps(rd, i0, conf)
                # gmf_max[i,:], gmf_dc[i,:], gmf_v[i,:], gmf_a[i,:], gmf_txp[i] = result
                # rgi = np.argmax(gmf_max[i,:])

            # """ 
            # """ ho=h5py.File(fname,"w")
            # ho["gmf"]=gmf_max
            # ho["gmf_dc"]=gmf_dc
            # ho["a"]=gmf_a
            # ho["v"]=gmf_v
            # ho["tx_pwr"]=gmf_txp
            # ho["i0"]=i0
            # ho.close() """
            # """

    logger.info(f"finishing job {job['idx']}/{job['N']}")
    return 100, {"outfile": outfile}