import logging
import numpy as np
import time
import h5py
from tqdm import tqdm
from pathlib import Path
from hardtarget.gmf import GMF_LIBS
from hardtarget import gmf_utils

from . import analysis_utils

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
):
    """
    Analyze data using gmf.

    TODO: update docstrings

    Parameters
    ----------
    rx: Digital_rf reader
    tx: Digital_rf reader
    gmf_config: path gfm config file
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
    rx_srcdir, rx_reader, rx_chnl = analysis_utils.load_source(rx)
    tx_srcdir, tx_reader, tx_chnl = analysis_utils.load_source(tx)

    # gmf params
    gmf_params = gmf_utils.load_gmf_params(rx_srcdir, config)
    # override config file
    if gmflib is not None:
        gmf_params["gmflib"] = gmflib
    logger.info("Using GMF backend: " + gmf_params["gmflib"])

    # bounds
    bounds = analysis_utils.compute_bounds(
        rx_reader,
        rx_chnl,
        gmf_params["sample_rate"],
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
    )

    # tasks
    total_tasks = analysis_utils.compute_total_tasks(gmf_params, bounds)
    job_tasks = analysis_utils.compute_job_tasks(job, total_tasks)
    tasks_skipped = 0

    logger.info(f"starting job {job['idx']}/{job['N']} with {len(job_tasks)} tasks")

    ipp = gmf_params["ipp"]
    ipp_samp = gmf_params["ipp_samp"]
    n_ipp = gmf_params["n_ipp"]
    num_cohints_per_file = gmf_params["num_cohints_per_file"]
    sample_rate = gmf_params["sample_rate"]

    # progress
    if progress:
        total = len(job_tasks)
        sub_desc_len = 8 + 2*len(str(num_cohints_per_file))
        progress_bar = tqdm(
            position=progress_position,
            desc="Processing".ljust(sub_desc_len, " ") if subprogress else "Processing",
            total=total,
        )

    # process
    results = {"dir": output, "files": [], "out": {}}
    for idx, task_idx in enumerate(job_tasks):
        # In case we are running MPI to distribute CUDA calculation
        # across multiple GPUs on multiple nodes we assume the ranks
        # distribute across the nodes in a linear fashion
        # e.g. node1[rank=(0,1,2)], node2[rank=(3,4,5)] for np=6 and node_GPUs=3
        gpu_id = task_idx % gmf_params["node_GPUs"]

        # initialise
        gmf_vals = []
        gmf_dc = []
        gmf_txp = []
        gmf_r_ind = []
        gmf_v_ind = []
        gmf_a_ind = []

        # filename outfile
        file_idx_sample = task_idx * ipp_samp * n_ipp * num_cohints_per_file + bounds[0]

        # filenames are in unix time microseconds
        epoch_unix = file_idx_sample / sample_rate
        epoch_unix_us = epoch_unix.astype("int64")*1000000
        filepath = analysis_utils.get_filepath(epoch_unix_us)

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
        r_vec = gmf_params["ranges"][r_inds]
        v_vec = gmf_params["range_rates"][gmf_v_ind[coh_ints, r_inds]]
        a_vec = gmf_params["accelerations"][gmf_a_ind[coh_ints, r_inds]]
        g_vec = gmf_vals[coh_ints, r_inds]

        if output is not None:
            # write result
            out = h5py.File(outfile, "w")
            gmf_axes = {}

            out["integration_index"] = np.arange(num_cohints_per_file, dtype=np.int64)
            gmf_axis = out["integration_index"]
            gmf_axis.make_scale("integration_index")
            gmf_axis.attrs["long_name"] = "Integration index within this file relative the epoch"
            gmf_axes["integration_index"] = gmf_axis

            grp = out.create_group("vector_params")
            for key in gmf_params:
                if key in gmf_utils.VECTOR_PARAM_KEYS:
                    grp[key] = gmf_params[key]
                elif key in gmf_utils.AXIS_PARAM_KEYS:
                    out[key] = gmf_params[key]
                    long_name, units = gmf_utils.AXIS_PARAM_KEYS[key]
                    gmf_axis = out[key]
                    gmf_axis.make_scale(key)
                    gmf_axis.attrs["long_name"] = long_name
                    gmf_axis.attrs["units"] = units
                    gmf_axes[key] = gmf_axis
                else:
                    if isinstance(gmf_params[key], bool):
                        val = np.bool_(gmf_params[key])
                    else:
                        val = gmf_params[key]
                    out.attrs[key] = val

            # TODO: this is if we want to attach scales to the dims
            # scales = [gmf_axes["integration_index"]]
            # if not gmf_params["reduce_range"]:
            #     scales.append(gmf_axes["ranges"])
            # if not gmf_params["reduce_range_rate"]:
            #     scales.append(gmf_axes["range_rates"])
            # if not gmf_params["reduce_acceleration"]:
            #     scales.append(gmf_axes["accelerations"])
            # per_int_scales = [gmf_axes["integration_index"]]

            analysis_utils.create_annotated_h5var(
                out, "gmf", gmf_vals,
                "Generalized Matched Filter output values",
            )
            analysis_utils.create_annotated_h5var(
                out, "gmf_zero_frequency", gmf_dc,
                "Range dependant noise floor (0-frequency gmf output)",
            )
            analysis_utils.create_annotated_h5var(
                out, "range_index", gmf_r_ind,
                "If range is reduced, contains the best range index for each left over axis",
            )
            analysis_utils.create_annotated_h5var(
                out, "range_rate_index", gmf_v_ind,
                "If range rate is reduced, contains the best range rate index for each left over axis",
            )
            analysis_utils.create_annotated_h5var(
                out, "acceleration_index", gmf_a_ind,
                "If acceleration is reduced, contains the best acceleration index for each left over axis",
            )

            analysis_utils.create_annotated_h5var(
                out, "range_peak", r_vec,
                "Range at peak GMF",
            )
            analysis_utils.create_annotated_h5var(
                out, "range_rate_peak", v_vec,
                "Range rate at peak GMF",
            )
            analysis_utils.create_annotated_h5var(
                out, "acceleration_peak", a_vec,
                "Acceleration at peak GMF",
            )
            analysis_utils.create_annotated_h5var(
                out, "gmf_peak", g_vec,
                "Peak GMF",
            )

            analysis_utils.create_annotated_h5var(
                out, "tx_power", gmf_txp,
                "Measured transmitted power",
            )
            analysis_utils.create_annotated_h5var(
                out, "epoch_unix", epoch_unix,
                "Epoch of first integration in unix time",
                units="s",
            )
            out.close()
            results["files"].append(filepath.name)
        else:
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

    # gmf lib
    gmf_lib_name = gmf_params.get("gmflib", None)
    if gmf_lib_name is None or gmf_lib_name not in GMF_LIBS:
        gmf_lib_name = "c" if "c" in GMF_LIBS else "numpy"
    gmf_lib = GMF_LIBS[gmf_lib_name]

    kwargs = {}
    if gmf_lib_name == "cuda":
        kwargs["gpu_id"] = gpu_id

    # TODO: implement this
    if not gmf_params["reduce_range_rate"] and gmf_params["reduce_acceleration"]:
        raise NotImplementedError("reduce settings not implemented")

    # parameters
    rx_stencil = gmf_params["rx_stencil"]
    tx_stencil = gmf_params["tx_stencil"]

    rx_reader, rx_channel = rx
    tx_reader, tx_channel = tx

    # read data vector with n_ipp + n_extra ipp's (to allow for searching across to subsequent pulses)
    z_rx = rx_reader.read_vector_1d(start_sample, gmf_params["read_length"], rx_channel)

    if tx_channel != rx_channel or tx_reader != rx_reader:
        z_tx = tx_reader.read_vector_1d(start_sample, gmf_params["read_length"], tx_channel)
    else:
        z_tx = np.copy(z_rx)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_rx = z_rx[rx_stencil]
    z_tx = z_tx[tx_stencil]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_amp2 = np.sum(np.abs(z_tx) ** 2.0)
    tx_amp = np.sqrt(tx_amp2)
    z_tx = np.conj(z_tx) / tx_amp

    # TODO: There are better ways of estimating the background noise by
    #   removing all coherent echoes first and using the individual signal samples

    gmf_vars = gmf_utils.GMFVariables(
        vals = np.zeros(gmf_params["gmf_size"], dtype=np.float32),
        dc = np.zeros([gmf_params["n_ranges"], ], dtype=np.float32),
        r_ind = np.full(gmf_params["gmf_size"], -1, dtype=np.int32),
        v_ind = np.full(gmf_params["gmf_size"], -1, dtype=np.int32),
        a_ind = np.full(gmf_params["gmf_size"], -1, dtype=np.int32),
    )

    if tx_amp > 1.0:
        gmf_lib(
            z_tx,
            z_rx,
            gmf_vars,
            gmf_params,
            **kwargs
        )

    return gmf_vars, tx_amp2
