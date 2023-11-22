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


####################################################################
# ANALYZE GMF
####################################################################

def _create_annotated_var(h5file, name, data, long_name, units=None):
    h5file[name] = data
    var = h5file[name]
    # TODO: this breaks vitables for some reason?
    #   but ncdump still recognizes the axis
    # for ind in range(len(scales)):
    #     var.dims[ind].attach_scale(scales[ind])
    var.attrs["long_name"] = long_name
    if units is not None:
        var.attrs["units"] = units


def analyze_gmf(
    rx,
    tx,
    config=None,
    start_time=None,
    end_time=None,
    relative_time=False,
    job={"idx": 1, "N": 1},
    logger=None,
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
    rx_srcdir, rx_reader, rx_chnl = load_source(rx)
    tx_srcdir, tx_reader, tx_chnl = load_source(tx)

    # gmf params
    gmf_params = gmf_utils.load_gmf_params(rx_srcdir, config)
    # override config file
    if gmflib is not None:
        gmf_params["gmflib"] = gmflib
    logger.info("Using GMF backend: " + gmf_params["gmflib"])

    # bounds
    bounds = compute_bounds(
        rx_reader,
        rx_chnl,
        gmf_params["sample_rate"],
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
    )

    # tasks
    total_tasks = compute_total_tasks(gmf_params, bounds)
    job_tasks = compute_job_tasks(job, total_tasks)
    tasks_skipped = 0

    # logging
    if logger is None:
        logger = logging.getLogger(__name__)
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
        filepath = get_filepath(epoch_unix_us)

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
        ts0 = time.time()
        for i in range(num_cohints_per_file):
            start_sample = file_idx_sample + i * ipp_samp * n_ipp
            gmf_vars, gmf_tx = analyze_ipps.analyze_ipps(
                (rx_reader, rx_chnl), (tx_reader, tx_chnl), start_sample, gmf_params
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

            _create_annotated_var(
                out, "gmf", gmf_vals,
                "Generalized Matched Filter output values",
            )
            _create_annotated_var(
                out, "gmf_zero_frequency", gmf_dc,
                "Range dependant noise floor (0-frequency gmf output)",
            )
            _create_annotated_var(
                out, "range_index", gmf_r_ind,
                "If range is reduced, contains the best range index for each left over axis",
            )
            _create_annotated_var(
                out, "range_rate_index", gmf_v_ind,
                "If range rate is reduced, contains the best range rate index for each left over axis",
            )
            _create_annotated_var(
                out, "acceleration_index", gmf_a_ind,
                "If acceleration is reduced, contains the best acceleration index for each left over axis",
            )

            _create_annotated_var(
                out, "range_peak", r_vec,
                "Range at peak GMF",
            )
            _create_annotated_var(
                out, "range_rate_peak", v_vec,
                "Range rate at peak GMF",
            )
            _create_annotated_var(
                out, "acceleration_peak", a_vec,
                "Acceleration at peak GMF",
            )
            _create_annotated_var(
                out, "gmf_peak", g_vec,
                "Peak GMF",
            )

            _create_annotated_var(
                out, "tx_power", gmf_txp,
                "Measured transmitted power",
            )
            _create_annotated_var(
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

            # out["range_peak"] = gmf_params["ranges"][gmf_r_ind]
            # out["range_rate_peak"] = gmf_params["range_rates"][gmf_v_ind]
            # out["acceleration_peak"] = gmf_params["accelerations"][gmf_a_ind]
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
