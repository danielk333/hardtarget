import datetime
from pathlib import Path
import numpy as np
import digital_rf as drf
from hardtarget import drf_utils
import h5py

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


####################################################################
# OUTPUT H5 FILE
####################################################################

VECTOR_PARAMS = [
    "rgs",
    "fvec",
    "acceleration_phasors",
    "rx_stencil",
    "tx_stencil",
    "rx_window_indices",
]


def create_annotated_h5var(h5file, name, data, long_name, units=None):
    """
    utility function for adding an attribute to a h5 file
    """
    h5file[name] = data
    var = h5file[name]
    # TODO: this breaks vitables for some reason?
    #   but ncdump still recognizes the axis
    # for ind in range(len(scales)):
    #     var.dims[ind].attach_scale(scales[ind])
    var.attrs["long_name"] = long_name
    if units is not None:
        var.attrs["units"] = units


####################################################################
# WRITE H5 FILE
####################################################################

# def toBool(val):
#    if isinstance(val, bool):
#        return np.bool_(val)

def write_h5_file(outfile,
                  gmf_params,
                  integration_ind,
                  gmf_vals,
                  gmf_dc,
                  gmf_txp,
                  gmf_r_ind,
                  gmf_v_ind,
                  gmf_a_ind,
                  r_vec,
                  v_vec,
                  a_vec,
                  g_vec,
                  epoch_unix):

    # write result
    out = h5py.File(outfile, "w")

    # EXPERIMENT PARAMS
    exp_grp = out.create_group("experiment")
    for key, val in gmf_params["EXP"].items():
        exp_grp[key] = val

    # GMF PROCESSING PARAMS
    pro_grp = out.create_group("processing")
    for key, val in gmf_params["PRO"].items():
        pro_grp[key] = val

    # derived parameters
    # der_grp = out.create_group("DER")
    # for key, val in gmf_params["DER"].items():
    #    der_grp[key] = val

    # VECTORS
    vector_grp = out.create_group("vector_params")
    for key in VECTOR_PARAMS:
        vector_grp[key] = gmf_params["DER"][key]

    # AXIS
    AXIS_MAP = {
        "integration_index": (
            integration_ind,
            "Integration index within this file relative the epoch",
            None),
        "ranges": (
            gmf_params["DER"]["ranges"],
            "Matched filter ranges",
            "m"),
        "range_rates": (
            gmf_params["DER"]["range_rates"],
            "Matched filter range rates",
            "m/s"),
        "accelerations": (
            gmf_params["DER"]["accelerations"],
            "Matched filter range accelerations",
            "m/s^2"),
    }

    for key, [val, long_name, units] in AXIS_MAP.items():
        out[key] = val
        _axis = out[key]
        _axis.make_scale(key)
        _axis.attrs["long_name"] = long_name
        if units is not None:
            _axis.attrs["units"] = units

    # TODO: this is if we want to attach scales to the dims
    # scales = [gmf_axes["integration_index"]]
    # if not gmf_params["reduce_range"]:
    #     scales.append(gmf_axes["ranges"])
    # if not gmf_params["reduce_range_rate"]:
    #     scales.append(gmf_axes["range_rates"])
    # if not gmf_params["reduce_acceleration"]:
    #     scales.append(gmf_axes["accelerations"])
    # per_int_scales = [gmf_axes["integration_index"]]

    # VARS
    VAR_MAP = {
        "gmf": (
            gmf_vals,
            "Generalized Matched Filter output values",
            None),
        "gmf_zero_frequency": (
            gmf_dc,
            "Range dependant noise floor (0-frequency gmf output)",
            None),
        "range_index": (
            gmf_r_ind,
            "If range is reduced, contains the best range index for each left over axis",
            None),
        "range_rate_index": (
            gmf_v_ind,
            "If range rate is reduced, contains the best range rate index for each left over axis",
            None),
        "acceleration_index": (
            gmf_a_ind,
            "If acceleration is reduced, contains the best acceleration index for each left over axis",
            None),
        # "range_peak": (r_vec, "Range at peak GMF", "m"),
        # "range_rate_peak": (v_vec, "Range rate at peak GMF", "m/s"),
        # "acceleration_peak": (a_vec, "Acceleration at peak GMF", "m/s^2"),
        "gmf_peak": (g_vec, "Peak GMF", None),
        "tx_power": (gmf_txp, "Measured transmitted power", "W"),
        "epoch_unix": (
            epoch_unix,
            "Epoch of first integration in unix time",
            "s"),
    }

    for key, (val, long_name, units) in VAR_MAP.items():
        create_annotated_h5var(out, key, val, long_name, units=units)

    out.close()
