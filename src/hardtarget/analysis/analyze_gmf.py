import logging
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from hardtarget.gmf import GMF_GRID_LIBS, GMF_OPTIMIZE_LIBS
from hardtarget.gmf_in_utils import load_gmf_params, GMFVariables
from hardtarget.gmf_out_utils import GMFOutArgs, dump_gmf_out

try:
    from mpi4py import MPI
except ImportError:
    comm = None
finally:
    comm = MPI.COMM_WORLD

from hardtarget.analysis import analysis_utils as a_utils

logger = logging.getLogger(__name__)


####################################################################
# ANALYZE GMF
####################################################################


def compute_gmf(
    rx,
    tx,
    config=None,
    start_time=None,
    end_time=None,
    relative_time=False,
    job={"idx": 1, "N": 1},
    progress=False,
    progress_position=0,
    subprogress=True,
    clobber=False,
    output=None,
    gmflib=None,
    gmf_optimize_lib=None,
):
    """
    Analyze data using gmf.

    TODO: update docstrings

    Parameters
    ----------
    rx: (drf source dir, channel name)
    tx: (drf source dir, channel name)
    config: path gfm config file
    job: used to identify subset of task indexes for this job

    Returns
    -------

    # TODO: make it able to return any portion of the GMF full map

    result: dict
        dir: string
            path to directory with files
        files: list
            paths to each generated file
        out: dictionary with in-memory results
    """

    # load data sources
    rx_srcdir, rx_reader, rx_chnl = a_utils.load_source(rx)
    tx_srcdir, tx_reader, tx_chnl = a_utils.load_source(tx)

    # gmf params
    gmf_params = load_gmf_params(rx_srcdir, config)
    params_exp = gmf_params["EXP"]
    params_pro = gmf_params["PRO"]
    params_der = gmf_params["DER"]

    # override config file
    # the gmflib parameter assumes the same lib source should be used for grid and fine tuning
    # unless explicitly set differently
    if gmflib is not None:
        params_pro["gmf_grid_lib"] = gmflib

    if gmf_optimize_lib is not None:
        params_pro["gmf_optimize_lib"] = gmf_optimize_lib
    elif gmflib is not None:
        params_pro["gmf_optimize_lib"] = gmflib
    if params_pro["gmf_optimize_lib"] == "no":
        params_pro["gmf_fine_tune"] = False

    logger.info("Using GMF grid backend: " + params_pro["gmf_grid_lib"])
    logger.info("Using GMF optimize backend: " + params_pro["gmf_optimize_lib"])

    # bounds
    bounds = a_utils.compute_bounds(
        rx_reader,
        rx_chnl,
        params_exp["sample_rate"],
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
    )

    # easy access
    ipp = params_exp["ipp"]
    sample_rate = params_exp["sample_rate"]
    n_ipp = params_pro["n_ipp"]
    num_cohints_per_file = params_pro["num_cohints_per_file"]
    ipp_samp = params_exp["ipp_samp"]

    # tasks
    total_tasks = a_utils.compute_total_tasks(ipp, n_ipp,
                                              num_cohints_per_file,
                                              bounds)
    job_tasks = a_utils.compute_job_tasks(job, total_tasks)
    tasks_skipped = 0

    logger.info(f"starting job {job['idx']}/{job['N']} with {len(job_tasks)} tasks")

    # progress
    if progress:
        total = len(job_tasks)
        sub_desc_len = 8 + 2*len(str(num_cohints_per_file))
        progress_bar = tqdm(
            position=progress_position,
            desc="Processing".ljust(sub_desc_len, " ") if subprogress else "Processing",
            total=total,
        )

    if comm is not None:
        # Make sure all jobs start at the same time
        # Good for logging, printing and debugging reasons as setup time should
        # still be minimal compared to the actual job times
        comm.barrier()

    # process
    results = {"dir": output, "files": [], "out": {}}
    for idx, task_idx in enumerate(job_tasks):
        # In case we are running MPI to distribute CUDA calculation
        # across multiple GPUs on multiple nodes we assume the ranks
        # distribute across the nodes in a linear fashion
        # e.g. node1[rank=(0,1,2)], node2[rank=(3,4,5)] for np=6 and node_gpus=3
        gpu_id = task_idx % params_pro["node_gpus"]

        # initialise
        gmf_vals = []
        gmf_dc = []
        gmf_txp = []
        gmf_r_ind = []
        gmf_v_ind = []
        gmf_a_ind = []
        peaks = []
        peak_vals = []
        # filename outfile
        file_idx_sample = task_idx * ipp_samp * n_ipp * num_cohints_per_file + bounds[0]

        # filenames are in unix time microseconds
        epoch_unix = file_idx_sample / sample_rate
        epoch_unix_us = epoch_unix.astype("int64")*1000000
        filepath = a_utils.get_filepath(epoch_unix_us)

        # write to file if output is defined
        if output is not None:
            outfile = Path(output) / filepath
            # crate directory
            dirname = Path(outfile).parent
            dirname.mkdir(parents=True, exist_ok=True)
            if outfile.is_file() and not clobber:
                logger.debug(
                    f"job {job['idx']}/{job['N']} already done task {idx}/{len(job_tasks)}"
                )
                tasks_skipped += 1
                continue

        # process
        # TODO: in case the RAM load is too heavy, this should write directly to disk instead
        # of collecting all the data
        ts0 = time.time()
        for i in range(num_cohints_per_file):
            start_sample = file_idx_sample + i * ipp_samp * n_ipp
            gmf_vars, gmf_tx = integrate_and_match_ipps(
                (rx_reader, rx_chnl),
                (tx_reader, tx_chnl),
                start_sample,
                gmf_params,
                gpu_id=gpu_id,
            )
            gmf_vals.append(gmf_vars.vals)
            gmf_dc.append(gmf_vars.dc)
            gmf_r_ind.append(gmf_vars.r_ind)
            gmf_v_ind.append(gmf_vars.v_ind)
            gmf_a_ind.append(gmf_vars.a_ind)
            peaks.append(gmf_vars.peak)
            peak_vals.append(gmf_vars.peak_val)
            gmf_txp.append(gmf_tx)

            if progress and subprogress:
                dots = ("."*(i % 4)).ljust(3, " ")
                total_num_len = len(str(num_cohints_per_file))
                curr_num = str(i + 1).ljust(total_num_len, ' ')
                progress_bar.set_description(f"Processing {curr_num}/{num_cohints_per_file} [{dots}]")
        ts1 = time.time()

        gmf_vals = np.stack(gmf_vals, axis=0)
        gmf_dc = np.stack(gmf_dc, axis=0)
        gmf_txp = np.stack(gmf_txp, axis=0)
        gmf_r_ind = np.stack(gmf_r_ind, axis=0)
        gmf_v_ind = np.stack(gmf_v_ind, axis=0)
        gmf_a_ind = np.stack(gmf_a_ind, axis=0)
        peaks = np.stack(peaks, axis=0)
        peak_vals = np.stack(peak_vals, axis=0)

        # log
        info = {
            "task": task_idx,
            "time": ts1 - ts0,
            "real": (ts1 - ts0) / (n_ipp * ipp * 1e-6 / sample_rate),
        }
        msg = "task_idx {task:4} time {time:1.2f} cpu/real {real:1.2f}"
        logger.debug(msg.format(**info))

        coh_ints = np.arange(num_cohints_per_file)

        r_inds = np.argmax(gmf_vals, axis=1)
        r_vec = params_der["ranges"][r_inds]
        v_vec = params_der["range_rates"][gmf_v_ind[coh_ints, r_inds]]
        a_vec = params_der["accelerations"][gmf_a_ind[coh_ints, r_inds]]
        g_vec = gmf_vals[coh_ints, r_inds]

        if output is not None:
            # DUMP TO FILE
            sample_numbers = np.arange(gmf_params["PRO"]["read_length"])

            gmf_out_args = GMFOutArgs(
                num_cohints_per_file=num_cohints_per_file,
                ranges=gmf_params["DER"]["ranges"],
                range_rates=gmf_params["DER"]["range_rates"],
                accelerations=gmf_params["DER"]["accelerations"],
                sample_numbers=sample_numbers,
                vals=gmf_vals,
                dc=gmf_dc,
                # r_ind=gmf_r_ind,
                v_ind=gmf_v_ind,
                a_ind=gmf_a_ind,
                txp=gmf_txp,
                r_vec=r_vec,
                v_vec=v_vec,
                a_vec=a_vec,
                g_vec=g_vec,
                peaks=peaks,
                peak_vals=peak_vals,
                rgs=gmf_params["DER"]["rgs"],
                fvec=gmf_params["DER"]["fvec"],
                decimated_sample_times=gmf_params["DER"]["decimated_sample_times"],
                acceleration_phasors=gmf_params["DER"]["acceleration_phasors"],
                rx_stencil=gmf_params["DER"]["rx_stencil"],
                tx_stencil=gmf_params["DER"]["tx_stencil"],
                rx_window_indices=gmf_params["DER"]["rx_window_indices"],
                epoch=epoch_unix)

            dump_gmf_out(gmf_out_args, gmf_params, outfile)
            results["files"].append(filepath.name)
        else:
            # write dict
            out = {}
            out["gmf"] = gmf_vals
            out["gmf_zero_frequency"] = gmf_dc
            out["range_index"] = gmf_r_ind
            out["range_rate_index"] = gmf_v_ind
            out["acceleration_index"] = gmf_a_ind
            out["range_peak"] = r_vec
            out["range_rate_peak"] = v_vec
            out["acceleration_peak"] = a_vec
            out["gmf_peak"] = g_vec
            out["tx_power"] = gmf_txp
            out["epoch_unix"] = epoch_unix
            results["data"][file_idx_sample] = out

        # progress
        if progress:
            progress_bar.update(1 + tasks_skipped)
            tasks_skipped = 0
    if progress:
        progress_bar.close()

    logger.info(f"finishing job {job['idx']}/{job['N']} with {len(job_tasks)} tasks")
    return results


####################################################################
# ANALYSE IPPS
####################################################################


def integrate_and_match_ipps(rx, tx, start_sample, gmf_params, gpu_id=0):
    """
    TODO: clean up this and all other docstrings when structure is done

    TODO: it would be nice if one could run this with signal data in the RAM
        instead of having to give or mock digitalrf readers

    Analyse ipps runs the gmf function.

    Parameters
    ----------
    rx : tuple
        (DigitalRF.reader, str) the drf reader and name of the channel with the rx signal
    tx : tuple
        (DigitalRF.reader, str) the drf reader and name of the channel with the tx signal
    start_sample : int
        Start sample of integration period
    gmf_params : dict
        GMF parameters

    Returns
    -------
    gmf_vars : hardtarget.gmf_utils.GMFVariables
        Named tuple container for compacting the variables set by the GMF function.
    tx_amp2 : float
        Squared summed tx amplitude
    """

    params_pro = gmf_params["PRO"]
    params_der = gmf_params["DER"]

    # gmf lib
    gmf_lib_name = params_pro.get("gmf_grid_lib", None)
    if gmf_lib_name is None:
        gmf_lib_name = "c" if "c" in GMF_GRID_LIBS else "numpy"
    elif gmf_lib_name not in GMF_GRID_LIBS:
        raise ValueError(
            f"Requested GMF gird lib '{gmf_lib_name}' not found in "
            "available libs, possible compilation errors in extensions\n"
            f"GMF_GRID_LIBS = {list(GMF_GRID_LIBS.keys())}"
        )
    gmf_lib = GMF_GRID_LIBS[gmf_lib_name]

    if gmf_params["PRO"].get("gmf_fine_tune", False):
        gmfo_lib_name = params_pro.get("gmf_optimize_lib", None)
        if gmfo_lib_name is None:
            gmfo_lib_name = "c" if "c" in GMF_OPTIMIZE_LIBS else "numpy"
        elif gmfo_lib_name not in GMF_OPTIMIZE_LIBS:
            raise ValueError(
                f"Requested GMF optimize lib '{gmfo_lib_name}' not found in "
                "available libs, possible compilation errors in extensions\n"
                f"GMF_OPTIMIZE_LIBS = {list(GMF_OPTIMIZE_LIBS.keys())}"
            )
        gmfo_lib = GMF_OPTIMIZE_LIBS[gmfo_lib_name]
    else:
        gmfo_lib = None

    kwargs = {}
    if gmf_lib_name == "cuda":
        kwargs["gpu_id"] = gpu_id

    # TODO: implement this
    if not params_pro["reduce_range_rate"] and params_pro["reduce_acceleration"]:
        raise NotImplementedError("reduce settings not implemented")

    # parameters
    rx_stencil = params_der["rx_stencil"]
    tx_stencil = params_der["tx_stencil"]

    rx_reader, rx_channel = rx
    tx_reader, tx_channel = tx

    # read data vector with n_ipp + n_extra ipp's (to allow for searching across to subsequent pulses)
    z_ipp = rx_reader.read_vector_1d(start_sample, params_pro["read_length"], rx_channel)

    if tx_channel != rx_channel or tx_reader != rx_reader:
        z_tx = tx_reader.read_vector_1d(start_sample, params_pro["read_length"], tx_channel)
    else:
        z_tx = np.copy(z_ipp)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_rx = z_ipp[rx_stencil].copy()
    z_tx = z_tx[tx_stencil]

    # TODO: generalize a preprocess filtering of 0 tx power
    # since it can cause unnessary slowdowns depending on experiment setup
    # e.g. a tx signal with multiple pulses with pauses between can be faster
    # computed by skipping the 0-tx periods inside the tx interval

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_amp2 = np.sum(np.abs(z_tx) ** 2.0)
    tx_amp = np.sqrt(tx_amp2)
    z_tx = np.conj(z_tx) / tx_amp

    gmf_vars = GMFVariables(
        vals = np.zeros(params_pro["gmf_size"], dtype=np.float32),
        dc = np.zeros(params_pro["n_ranges"], dtype=np.float32),
        r_ind = np.full(params_pro["gmf_size"], -1, dtype=np.int32),
        v_ind = np.full(params_pro["gmf_size"], -1, dtype=np.int32),
        a_ind = np.full(params_pro["gmf_size"], -1, dtype=np.int32),
        peak = np.full((3,), np.nan, dtype=np.float32),
        peak_val = np.full((1,), np.nan, dtype=np.float32),
    )

    if tx_amp > 1.0:
        gmf_lib(
            z_tx,
            z_rx,
            gmf_vars,
            gmf_params,
            **kwargs
        )
        if gmfo_lib is not None:
            max_ind = np.argmax(np.abs(gmf_vars.vals))
            gmf_start = np.array([
                gmf_params["DER"]["ranges"][max_ind],
                gmf_params["DER"]["range_rates"][gmf_vars.v_ind[max_ind]],
                gmf_params["DER"]["accelerations"][gmf_vars.a_ind[max_ind]],
            ])
            opt_result = gmfo_lib(
                z_tx,
                z_ipp,
                gmf_params,
                gmf_start,
            )
            gmf_vars.peak[:] = opt_result.x
            gmf_vars.peak_val[0] = opt_result.fun

    return gmf_vars, tx_amp2
