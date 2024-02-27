import logging
import numpy as np
import time
import h5py
from tqdm import tqdm
from pathlib import Path
from hardtarget.gmf import get_avalible_libs, get_estimation_method, MethodType
from hardtarget.configuration import load_gmf_params
from . import utils


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
    job=None,
    progress=False,
    progress_position=0,
    subprogress=True,
    clobber=False,
    output=None,
    gmf_method=None,
    gmf_implementation=None,
    logger=None
):
    """
    Analyze data using gmf.

    TODO: update docstrings

    Parameters
    ----------
    :parameter rx: (drf source dir, channel name)
    :parameter tx: (drf source dir, channel name)
    :parameter config: path gfm config file
    :parameter job: used to identify subset of task indexes for this job

    Returns
    -------

    # TODO: this function needs to be cleaned up a bit

    :return: dict
        dir: string
            path to directory with files
        files: list
            paths to each generated file
        out: dictionary with in-memory results

    """

    if job is None:
        job = {"idx": 0, "N": 1}

    if logger is None:
        logger = logging.getLogger(__name__)

    # load data sources
    rx_srcdir, rx_reader, rx_chnl = utils.load_source(rx)
    tx_srcdir, tx_reader, tx_chnl = utils.load_source(tx)

    # gmf params
    gmf_params = load_gmf_params(rx_srcdir, config)
    params_exp = gmf_params["EXP"]
    params_pro = gmf_params["PRO"]
    params_der = gmf_params["DER"]

    # override config file with cli arguments
    if gmf_method is not None:
        params_pro["gmf_method"] = gmf_method
    else:
        gmf_method = params_pro["gmf_method"]
    if gmf_implementation is not None:
        params_pro["gmf_implementation"] = gmf_implementation
    else:
        gmf_implementation = params_pro["gmf_implementation"]

    lib, libtype = get_estimation_method(
        params_pro["gmf_implementation"],
        params_pro["gmf_method"],
    )

    if lib is None:
        raise ValueError(
            f"Cannot find requested method '{gmf_method}' "
            f"in requested implementation '{gmf_implementation}'\n"
            f"Avalible implemented methods: \n{get_avalible_libs(indent=' '*4)}"
        )
    lib_kwargs = {}
    # In case we are running MPI to distribute CUDA calculation
    # across multiple GPUs on multiple nodes we assume the ranks
    # distribute across the nodes in a linear fashion
    # e.g. node1[rank=(0,1,2)], node2[rank=(3,4,5)] for np=6 and node_gpus=3
    # So gpuid = rank % node_gpus = node1[gpuid=(0,1,2)], node2[gpuid=(0,1,2)]
    if gmf_implementation == "cuda":
        lib_kwargs["gpu_id"] = job["idx"] % params_pro["node_gpus"]

    logger.info(f"Using GMF method {gmf_method} ({gmf_implementation})")

    # bounds
    bounds = utils.compute_bounds(
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
    total_tasks = utils.compute_total_tasks(
        ipp, n_ipp,
        num_cohints_per_file,
        bounds,
    )

    job_tasks = utils.compute_job_tasks(job, total_tasks)
    job_cohints = (len(job_tasks) - 2) * num_cohints_per_file

    num_cohints = num_cohints_per_file
    file_idx_sample = np.max(job_tasks) * ipp_samp * n_ipp * num_cohints_per_file + bounds[0]
    if file_idx_sample + num_cohints_per_file * ipp_samp * n_ipp > bounds[1]:
        num_cohints = int((bounds[1] - file_idx_sample) // (ipp_samp * n_ipp))
    job_cohints += num_cohints
    num_cohints = num_cohints_per_file
    file_idx_sample = np.min(job_tasks) * ipp_samp * n_ipp * num_cohints_per_file + bounds[0]
    if file_idx_sample < bounds[0]:
        num_cohints = num_cohints_per_file - int((bounds[0] - file_idx_sample) // (ipp_samp * n_ipp))
    job_cohints += num_cohints

    tasks_skipped = 0

    logger.info(f"starting job {job['idx']}/{job['N']} with {len(job_tasks)} tasks")

    # progress
    progress_desc = "Coherent integrations"
    if progress:
        total = job_cohints
        extend_str_len = len(str(len(job_tasks)))
        total_num = str(len(job_tasks)).ljust(extend_str_len, " ")
        curr_num = "1".ljust(extend_str_len, " ")
        subprog_str = f"[file {curr_num}/{total_num}]"
        progress_bar = tqdm(
            position=progress_position,
            desc=f"{progress_desc} {subprog_str}" if subprogress else progress_desc,
            total=total,
        )

    # process
    results = {"dir": output, "files": [], "data": {}}
    for idx, task_idx in enumerate(job_tasks):

        # initialise
        all_gmf_vars = []
        # filename outfile
        file_idx_sample = task_idx * ipp_samp * n_ipp * num_cohints_per_file + bounds[0]

        # filenames are in unix time microseconds
        epoch_unix_us = file_idx_sample / (sample_rate/1000000)
        epoch_unix = epoch_unix_us.astype("float128")/1000000
        epoch_unix_us = epoch_unix_us.astype("int64")
        filepath = utils.get_filepath(epoch_unix_us)

        # write to file if output is defined
        if output is not None:
            outfile = Path(output) / filepath
            # crate directory
            dirname = Path(outfile).parent
            dirname.mkdir(parents=True, exist_ok=True)

            # if-statement from hell
            if (
                # if exist and not override, check if calculation is already done
                (outfile.is_file() and not clobber)
                and (
                    # calculation is always already done for grid
                    libtype == MethodType.grid
                    or (
                        # calculation is done for optimize if the variables exist
                        libtype == MethodType.optimize
                        and utils.check_for_optimize_result(outfile)
                    )
                )
            ):
                logger.debug(
                    f"job {job['idx']}/{job['N']} already done task {idx}/{len(job_tasks)}"
                )
                if progress:
                    progress_bar.update(num_cohints_per_file)
                tasks_skipped += 1
                continue
        if libtype == MethodType.optimize:
            # TODO: this should be taken from a configurable variable in the h5 file
            # this would allow for a inter-mediate step to actually go in and
            # use the data from the entire event to inform good starting values
            # for even points that failed to be analyzed with the grid method
            # (since this has higher sensitivity)
            # TODO: this should have a "do-calculation" vector loaded too where
            # optimization can be skiped where no event was detection in the rough analysis
            #
            # Then the work-flow would be
            # 1) run hardtarget with a grid method
            # 2) run resordan to do tresholding / clustering and set start
            #    values and skip empty portions of data
            # 3) run hardtarget with an optimize method to refine events and get optimal SNR
            # 4) run resordan to create event files
            with h5py.File(outfile, "r") as hf:
                gmf_starts = np.stack([
                    hf["range_peak"][()],
                    hf["range_rate_peak"][()],
                    hf["acceleration_peak"][()],
                ])

        # process
        # TODO: in case the RAM load is too heavy, this should write directly to disk instead
        # of collecting all the data
        ts0 = time.time()
        num_cohints = num_cohints_per_file
        if file_idx_sample + num_cohints_per_file * ipp_samp * n_ipp > bounds[1]:
            num_cohints = int((bounds[1] - file_idx_sample) // (ipp_samp * n_ipp))
        start_cohind = 0
        # TODO: until this is fixed, adaptive start interval disabled
        # TODO: fix, this indexing gets messed up in the optimize step, need to look over
        # if file_idx_sample < bounds[0]:
        #     start_cohind = int((bounds[0] - file_idx_sample) // (ipp_samp * n_ipp))

        for coh_ind in range(start_cohind, num_cohints):
            start_sample = file_idx_sample + coh_ind * ipp_samp * n_ipp
            if progress:
                progress_bar.update(1)
            if libtype == MethodType.grid:
                gmf_vars = grid_integrate_and_match_ipps(
                    rx=(rx_reader, rx_chnl),
                    tx=(tx_reader, tx_chnl),
                    start_sample=start_sample,
                    gmf_params=gmf_params,
                    gmf_lib=lib,
                    lib_kwargs=lib_kwargs,
                )
                all_gmf_vars.append(gmf_vars)
            elif libtype == MethodType.optimize:
                gmf_vars = optimize_integrate_and_match_ipps(
                    rx=(rx_reader, rx_chnl),
                    tx=(tx_reader, tx_chnl),
                    start_sample=start_sample,
                    gmf_start=gmf_starts[:, coh_ind],
                    gmf_params=gmf_params,
                    gmf_lib=lib,
                    lib_kwargs=lib_kwargs,
                )
                all_gmf_vars.append(gmf_vars)

        ts1 = time.time()

        all_gmf_vars = utils.stack_gmf_vars(all_gmf_vars, libtype)

        # log
        info = {
            "task": task_idx,
            "time": ts1 - ts0,
            "real": (ts1 - ts0) / (n_ipp * ipp * 1e-6 / sample_rate),
        }
        msg = "task_idx {task:4} time {time:1.2f} cpu/real {real:1.2f}"
        logger.debug(msg.format(**info))

        if libtype == MethodType.grid:
            sample_numbers = np.arange(gmf_params["PRO"]["read_length"])
            coh_ints = np.arange(num_cohints)

            r_inds = np.argmax(all_gmf_vars.vals, axis=1)
            r_vec = params_der["ranges"][r_inds]
            v_vec = params_der["range_rates"][all_gmf_vars.v_ind[coh_ints, r_inds]]
            a_vec = params_der["accelerations"][all_gmf_vars.a_ind[coh_ints, r_inds]]
            g_vec = all_gmf_vars.vals[coh_ints, r_inds]

            gmf_out_args = utils.GMFOutArgs(
                num_cohints_per_file=num_cohints,
                ranges=gmf_params["DER"]["ranges"],
                range_rates=gmf_params["DER"]["range_rates"],
                accelerations=gmf_params["DER"]["accelerations"],
                sample_numbers=sample_numbers,
                vals=all_gmf_vars.vals,
                dc=all_gmf_vars.dc,
                v_ind=all_gmf_vars.v_ind,
                a_ind=all_gmf_vars.a_ind,
                txp=all_gmf_vars.tx_pwr,
                r_vec=r_vec,
                v_vec=v_vec,
                a_vec=a_vec,
                g_vec=g_vec,
                epoch=epoch_unix,
            )
        elif libtype == MethodType.optimize:
            gmf_out_args = utils.GMFOptimizeOutArgs(
                peaks=all_gmf_vars.peak,
                peak_vals=all_gmf_vars.peak_val,
            )

        if output is not None:
            # DUMP TO FILE

            # TODO: this might actually be a whole lot cleaner with just a class that can carry
            # the definitions, code, formatting, appending, ect with it
            # Then this if else thing is replaced by the fact that the class instance is different
            # with different implementation for the same method, extracting out common functions
            # outside the class
            if libtype == MethodType.grid:
                utils.dump_gmf_out(gmf_out_args, gmf_params, outfile, mode="w", meta=True)
            elif libtype == MethodType.optimize:
                utils.dump_gmf_out(gmf_out_args, gmf_params, outfile, clobber=clobber, mode="a", meta=False)
            results["files"].append(filepath.name)
        else:
            # write dict
            if isinstance(gmf_out_args, utils.GMFOutArgs):
                data_variables = utils.define_grid_variables(gmf_out_args).items()
            elif isinstance(gmf_out_args, utils.GMFOptimizeOutArgs):
                data_variables = utils.define_optimize_variables(gmf_out_args).items()

            results["data"][file_idx_sample] = data_variables

        # progress
        if progress:
            curr_num = f"{idx + 1}".ljust(extend_str_len, " ")
            subprog_str = f"[file {curr_num}/{total_num}]"
            progress_bar.set_description(
                f"{progress_desc} {subprog_str}" if subprogress else progress_desc,
            )
            tasks_skipped = 0
    if progress:
        progress_bar.close()

    logger.info(f"finishing job {job['idx']}/{job['N']} with {len(job_tasks)} tasks")
    return results


####################################################################
# ANALYSE IPPS
####################################################################


def extract_signals(rx, tx, start_sample, gmf_params):
    """Extract the signals needed to analyse the data

    TODO: docstring
    """

    params_pro = gmf_params["PRO"]
    params_der = gmf_params["DER"]

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

    return z_tx, z_rx, z_ipp


def grid_integrate_and_match_ipps(rx, tx, start_sample, gmf_params, gmf_lib, lib_kwargs):
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
    gmf_vars : hardtarget.analysis.utils.GMFVariables
        Named tuple container for compacting the variables set by the GMF function.
    tx_amp2 : float
        Squared summed tx amplitude
    """

    z_tx, z_rx, z_ipp = extract_signals(rx, tx, start_sample, gmf_params)
    # TODO: generalize a preprocess filtering of 0 tx power
    # since it can cause unnessary slowdowns depending on experiment setup
    # e.g. a tx signal with multiple pulses with pauses between can be faster
    # computed by skipping the 0-tx periods inside the tx interval

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_pwr = np.sum(np.abs(z_tx) ** 2.0)
    tx_amp = np.sqrt(tx_pwr)
    z_tx = np.conj(z_tx) / tx_amp

    size = (gmf_params["PRO"]["n_ranges"], )
    gmf_vars = utils.GMFVariables(
        vals = np.zeros(size, dtype=np.float32),
        dc = np.zeros(size, dtype=np.float32),
        v_ind = np.full(size, -1, dtype=np.int32),
        a_ind = np.full(size, -1, dtype=np.int32),
        tx_pwr = tx_pwr,
    )

    if tx_amp > 1.0:
        gmf_lib(
            z_tx,
            z_rx,
            gmf_vars,
            gmf_params,
            **lib_kwargs
        )

    return gmf_vars


def optimize_integrate_and_match_ipps(rx, tx, start_sample, gmf_start, gmf_params, gmf_lib, lib_kwargs):
    """
    TODO: do this docstring

    """

    z_tx, z_rx, z_ipp = extract_signals(rx, tx, start_sample, gmf_params)

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_pwr = np.sum(np.abs(z_tx) ** 2.0)
    tx_amp = np.sqrt(tx_pwr)
    z_tx = np.conj(z_tx) / tx_amp

    gmf_vars = utils.GMFOptimizeVariables(
        peak = np.zeros((3, ), dtype=np.float64),
        peak_val = np.zeros((1, ), dtype=np.float64),
    )

    if tx_amp > 1.0:
        gmf_vars.peak[:], gmf_vars.peak_val[0] = gmf_lib(
            z_tx,
            z_ipp,
            gmf_params,
            gmf_start,
            **lib_kwargs
        )

    return gmf_vars
