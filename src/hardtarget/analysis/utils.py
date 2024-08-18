import datetime
from pathlib import Path
import numpy as np
import digital_rf as drf
import h5py
from collections import namedtuple
import re
import logging

from hardtarget.gmf import MethodType
from hardtarget import drf_utils
import hardtarget.digitalrf_wrapper as drf_wrapper

logger = logging.getLogger(__name__)

####################################################################
# GMF OUTPUT DATA
####################################################################

"""Container for compacting the variables set by the GMF Grid function."""
GMFVariables = namedtuple(
    "GMFVariables",
    [
        "vals",  # match function values reduced over the requested axis
        "dc",  # 0-frequency gmf output as a function of range
        "v_ind",  # best fitting range-rate
        "a_ind",  # best fitting range-rate change
        "tx_pwr",  # tx power
    ],
)

"""Container for compacting the variables set by the GMF Optimize function."""
GMFOptimizeVariables = namedtuple(
    "GMFOptimizeVariables",
    [
        "peak",  # peak location
        "peak_val",  # peak magnitude
    ],
)


def stack_gmf_vars(gmf_vars_list, libtype):
    if libtype == MethodType.grid:
        return GMFVariables(
            vals = np.stack([x.vals for x in gmf_vars_list], axis=0),
            dc = np.stack([x.dc for x in gmf_vars_list], axis=0),
            v_ind = np.stack([x.v_ind for x in gmf_vars_list], axis=0),
            a_ind = np.stack([x.a_ind for x in gmf_vars_list], axis=0),
            tx_pwr = np.stack([x.tx_pwr for x in gmf_vars_list], axis=0),
        )
    elif libtype == MethodType.optimize:
        return GMFOptimizeVariables(
            peak = np.stack([x.peak for x in gmf_vars_list], axis=0),
            peak_val = np.stack([x.peak_val for x in gmf_vars_list], axis=0),
        )


################################################################
# LOAD GMF OUT (H5)
#
# Parse groups and datasets recursively
################################################################

def check_for_optimize_result(file):
    with h5py.File(file, "r") as hf:
        check = "gmf_optimize" in hf
    return check


def collect_gmf_data(paths, mats=None, vecs=None):
    """Given a number of input h5 GMF paths, collect and concatenate some of the useful data.
    """
    data = None
    meta = None
    if mats is None:
        # Default mats
        mats = [
            "gmf",
            "gmf_optimized_peak",
            "gmf_zero_frequency",
            "range_rate_index",
            "acceleration_index",
            "nf_vec",
        ]
    if vecs is None:
        # Default vecs
        vecs = [
            "range_peak",
            "range_rate_peak",
            "acceleration_peak",
            "gmf_optimized",
            "gmf_peak",
            "tx_power",
            "t"
        ]
    derived = ["t", "nf_vec"]
    optional = ["gmf_optimized_peak", "gmf_optimized"]

    mats_data = {}
    vecs_data = {}
    for path in paths:
        with h5py.File(path, "r") as hf:
            for key in mats:
                if key in derived:
                    continue
                if key not in hf and key in optional:
                    continue
                mats_data[key] = hf[key][()]
            for key in vecs:
                if key in derived:
                    continue
                if key not in hf and key in optional:
                    continue
                vecs_data[key] = hf[key][()]

            if meta is None:
                meta = {
                    "ranges": hf["ranges"][()],
                    "range_rates": hf["range_rates"][()],
                    "accelerations": hf["accelerations"][()],
                    "range_gates": np.arange(
                        hf["processing"].attrs["min_range_gate"],
                        hf["processing"].attrs["min_range_gate"],
                    ),
                }
                meta["processing"] = {key: val for key, val in hf["processing"].attrs.items()}
                meta["experiment"] = {key: val for key, val in hf["experiment"].attrs.items()}
                for key, val in hf.attrs.items():
                    meta[key] = val
            epoch_unix = hf["epoch_unix"][()]
            _t_conv = (hf["processing"].attrs["n_ipp"][()] * hf["experiment"].attrs["ipp"][()]) * 1e-6

        # Additional useful parameters to calculate
        n_cohints = mats_data["gmf"].shape[0]
        vecs_data["t"] = (np.arange(n_cohints) + 1) * _t_conv + epoch_unix
        nf_vec = np.nanmedian(mats_data["gmf_zero_frequency"], axis=0)
        nf_vec = nf_vec.reshape((1, nf_vec.size))
        mats_data["nf_vec"] = nf_vec

        if data is None:
            data = {}
            for key in mats_data:
                logger.debug(f"Init mat {key}: {mats_data[key].shape} [{mats_data[key].dtype}]")
                data[key] = mats_data[key]
            for key in vecs_data:
                logger.debug(f"Init vec {key}: {vecs_data[key].shape} [{vecs_data[key].dtype}]")
                data[key] = vecs_data[key]
        else:
            for key in mats_data:
                logger.debug(f"Append mat {key}: {mats_data[key].shape} [{mats_data[key].dtype}]")
                data[key] = np.append(data[key], mats_data[key], axis=0)
            for key in vecs_data:
                logger.debug(f"Append vec {key}: {vecs_data[key].shape} [{vecs_data[key].dtype}]")
                data[key] = np.append(data[key], vecs_data[key])

    data["nf_range"] = np.nanmedian(data["nf_vec"], axis=0)

    return data, meta


def yield_chunked_gmf_data(paths, chunk_size=None):
    paths.sort()
    pth_num = len(paths)
    if chunk_size is None:
        chunks = 1
        chunk_size = pth_num
    else:
        chunks = pth_num // chunk_size + 1
    for ind in range(chunks):
        sub_paths = paths[(ind * chunk_size):((ind + 1) * chunk_size)]
        yield collect_gmf_data(sub_paths)


def all_gmf_h5_files(gmf_folder):
    """generate all files matching 'yyyy-mm-ddThh-00-00/gmf-*.h5'"""
    top = Path(gmf_folder)
    dir_pattern = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}-00-00')
    subdirs = [d for d in top.iterdir() if d.is_dir() and dir_pattern.match(d.name)]
    file_pattern = re.compile(r'^gmf-.*\.h5$')
    files = []
    for subdir in subdirs:
        files += [f for f in subdir.iterdir() if f.is_file and file_pattern.match(f.name)]
    return files


def collect_gmf_paths(
    folder,
    start_time=None,
    end_time=None,
    relative_time=False,
):
    fl = all_gmf_h5_files(folder)
    fl.sort()
    fl_epochs = [int(file.stem.split("-")[1]) * 1e-6 for file in fl]

    epoch_unix = fl_epochs[0]
    max_unix = fl_epochs[-1]

    if relative_time:
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = max_unix - epoch_unix
        unix_t0 = epoch_unix + start_time
        unix_t1 = epoch_unix + end_time
    else:
        if start_time is None:
            start_time = datetime.datetime.utcfromtimestamp(epoch_unix)
        if end_time is None:
            end_time = datetime.datetime.utcfromtimestamp(max_unix)

        dt64_t0 = start_time if isinstance(start_time, np.datetime64) else np.datetime64(start_time)
        unix_t0 = dt64_t0.astype("datetime64[us]").astype("int64") * 1e-6

        dt64_t1 = end_time if isinstance(end_time, np.datetime64) else np.datetime64(end_time)
        unix_t1 = dt64_t1.astype("datetime64[us]").astype("int64") * 1e-6

    fl = [file for file, ep in zip(fl, fl_epochs) if ep >= unix_t0 and ep <= unix_t1]

    # TODO - provide useful feedback if only one file exists
    # and start_time, end_time is given, but not a perfect match with the file

    return fl


def is_scale(obj):
    return obj.attrs.get("CLASS") == b"DIMENSION_SCALE"


def inspect_h5_node(obj, path=[]):
    items = []
    for child_key, child_item in obj.items():
        child_path = path + [child_key]
        if isinstance(child_item, h5py.Group):
            items += inspect_h5_node(child_item, child_path)
        else:
            items.append(inspect_h5_leaf(child_item, child_path))
    items.append(inspect_h5_attributes(obj, path))
    return items


def inspect_h5_attributes(obj, path):
    item = {
        "type": "attributes",
        "attrs": {key: val for key, val in obj.attrs.items()},
    }
    return path, item


def inspect_h5_leaf(obj, path):
    item = {}

    if isinstance(obj, h5py.Dataset):
        item["scale"] = is_scale(obj)
        item["dtype"] = obj.dtype
        item["value"] = obj[()]

        if obj.dtype == "object":
            item["type"] = "object"
        elif obj.shape == ():
            item["type"] = "scalar"
        else:
            item["type"] = "dataset"
            item["shape"] = obj.shape
    else:
        item["type"] = "other"
        item["value"] = obj

    return path, item


def load_gmf_out(
    folder,
    start_time=None,
    end_time=None,
    relative_time=False,
    chunk_size=None,
):
    """Create a generator that loads and concatenates all GMF output data from
    given folder between the given time periods in desired chunks.
    """
    files = collect_gmf_paths(
        folder,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
    )

    return yield_chunked_gmf_data(files, chunk_size=chunk_size)


####################################################################
# DUMP GMF OUT
####################################################################


def dump_gmf_out(gmf_out_args, gmf_params, outfile, clobber=False, mode="w", meta=True):
    out = h5py.File(outfile, mode)

    if isinstance(gmf_out_args, GMFOutArgs):
        data_variables = define_grid_variables(gmf_out_args).items()
    elif isinstance(gmf_out_args, GMFOptimizeOutArgs):
        data_variables = define_optimize_variables(gmf_out_args).items()

    # VARIABLES
    for key, item in data_variables:
        # scale
        is_scale = "scale" in item and item["scale"]

        # set group as target (only non-scales)
        target = out
        if "group" in item and not is_scale:
            grp_name = item["group"]
            if grp_name not in out:
                out.create_group(grp_name)
            target = out[grp_name]
        # create dataset
        if clobber and key in target:
            del target[key]
        ds = target.create_dataset(key, data=item["data"])
        # register scale
        if is_scale:
            ds.make_scale(key)
        # attach ds dimensions to scales
        if "dims" in item:
            for idx, (scale_key, label) in enumerate(item["dims"]):
                scale = out[scale_key]
                ds.dims[idx].attach_scale(scale)
                ds.dims[idx].label = label
        # set name and units
        ds.attrs["long_name"] = item["long_name"]
        if "units" in item:
            ds.attrs["units"] = item["units"]

    if meta:
        # TODO: these should be saved as variables so they can be documented
        # EXPERIMENT PARAMS
        if "experiment" not in out:
            out.create_group("experiment")
        exp_grp = out["experiment"]
        for key, val in gmf_params["EXP"].items():
            exp_grp.attrs[key] = val

        # GMF PROCESSING PARAMS
        if "processing" not in out:
            out.create_group("processing")
        pro_grp = out["processing"]
        for key, val in gmf_params["PRO"].items():
            pro_grp.attrs[key] = val

        # EPOCH
        out["epoch_unix"] = gmf_out_args.epoch

    out.close()


####################################################################
# GMF OUT ARGUMENTS
####################################################################


"""Container for compacting the variables set by the GMF function."""
GMFOutArgs = namedtuple(
    "GMFOutArgs",
    [
        "num_cohints_per_file",
        "ranges",
        "range_rates",
        "accelerations",
        "sample_numbers",
        "vals",
        "dc",
        "v_ind",
        "a_ind",
        "txp",
        "r_vec",
        "v_vec",
        "a_vec",
        "g_vec",
        "pointing_vec",
        "epoch"
    ],
)

GMFOptimizeOutArgs = namedtuple(
    "GMFOptimizeOutArgs",
    [
        "peaks",
        "peak_vals",
    ],
)


####################################################################
# H5 VARIABLE DEFINITIONS
####################################################################


def define_grid_variables(gmf_out_args):
    integration_index = np.arange(gmf_out_args.num_cohints_per_file, dtype=np.int64)

    return {
        "integration_index": {
            "data": integration_index,
            "long_name": "Integration index within this file relative the file epoch",
            "scale": True
        },
        "ranges": {
            "data": gmf_out_args.ranges,
            "long_name": "Matched filter ranges",
            "units": "m",
            "scale": True
        },
        "range_rates": {
            "data": gmf_out_args.range_rates,
            "long_name": "Matched filter range rates",
            "units": "m/s",
            "scale": True
        },
        "accelerations": {
            "data": gmf_out_args.accelerations,
            "long_name": "Matched filter range accelerations",
            "units": "m/s^2",
            "scale": True
        },
        "sample_numbers": {
            "data": gmf_out_args.sample_numbers,
            "long_name": "Receiver sample number in radar cycle",
            "scale": True
        },
        "gmf": {
            "data": gmf_out_args.vals,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": "Generalized Matched Filter output values",
        },
        "gmf_zero_frequency": {
            "data": gmf_out_args.dc,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": "Range dependant noise floor (0-frequency gmf output)",
        },
        "range_rate_index": {
            "data": gmf_out_args.v_ind,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": (
                "If range_rate is reduced, contains the best range rate index "
                "for each left over axis"),
        },
        "acceleration_index": {
            "data": gmf_out_args.a_ind,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": (
                "If acceleration is reduced, contains the best acceleration "
                "index for each left over axis"),
        },
        "tx_power": {
            "data": gmf_out_args.txp,
            "dims": [("integration_index", "t")],
            "long_name": "Transmitted signal power",
        },
        "range_peak": {
            "data": gmf_out_args.r_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Range at peak GMF",
        },
        "range_rate_peak": {
            "data": gmf_out_args.v_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Range rate at peak GMF",
        },
        "acceleration_peak": {
            "data": gmf_out_args.a_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Acceleration at peak GMF",
        },
        "gmf_peak": {
            "data": gmf_out_args.g_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Peak GMF",
        },
        "pointing": {
            "data": gmf_out_args.pointing_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Radar pointing data (azimuth, elevation)"
        }
    }


def define_optimize_variables(gmf_out_args):
    return {
        "gmf_optimized_peak": {
            "data": gmf_out_args.peaks,
            "dims": [("integration_index", "t")],
            "long_name": "Fine tuned range, range-rate and acceleration",
            # "group": "gmf"
        },
        "gmf_optimized": {
            "data": gmf_out_args.peak_vals,
            "dims": [("integration_index", "t")],
            "long_name": "Generalized Matched Filter fine tuned peak output values",
            # "group": "gmf"
        },
    }


####################################################################
# DIGITAL RF READERS
####################################################################


def is_pair_unpackable(a):
    """Returns true if a can be unpacked to a pair of variables."""
    return True if isinstance(a, (tuple, list)) and len(a) == 2 else False

def load_source(src):
    """
    Load Digital_rf reader object from (srcdir, chnl)
    """
    if type(src) is not tuple:
        raise ValueError(f"tuple (path, chnl) expected, {src}")
    if not is_pair_unpackable(src):
        raise ValueError(f"tuple (path, chnl) expected, {src}")
    path, chnl = src
    if path is None or not Path(path).is_dir():
        raise ValueError(f"path must be directory, {path}")
    # read cached instance
    reader = drf.DigitalRFReader([str(path)])
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
    >>> get_tasks({"idx":1, "N":2}, 8)
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
    n_tasks = np.ceil(np.floor((bounds[1] - bounds[0]) / (ipp * n_ipp)) / num_cohints_per_file).astype(int)
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
# LOAD POINTING DATA
####################################################################


def ts_from_index(idx, sample_rate, ts_origin=0):
    """convert from sample idx to timestamp"""
    return (idx / float(sample_rate)) + ts_origin

def index_from_ts(ts, sample_rate, ts_origin=0):
    """convert from timestamp to sample index"""
    return int((ts-ts_origin) * sample_rate)

def load_metadata(reader, interval, target_rate, target_value):
    """
    read metadata data for time interval and upsample to sample_rate
    
    Parameters
    ----------
    reader: digitalrf_wrapper.DigitalMetadataReader
        reader object for metadata stream
    interval: [float, float]
        timestamps in seconds since epoch
    target_rate: float
        upsample metadata according to given target_rate
    target_value(item): function
        return value for metadata item

    Returns
    -------
    numpy.ndarray
        float array of values
    """
    # query metadata
    idx_start = index_from_ts(interval[0], reader.sample_rate)
    idx_end = index_from_ts(interval[1], reader.sample_rate) + 1
    result = reader.read(idx_start, idx_end)
    metadata_indexes, metadata_items = zip(*result)

    # convert metadata indexes to target rate indexes 
    def convert(metadata_idx):
        ts = ts_from_index(metadata_idx, reader.sample_rate)
        return index_from_ts(ts, target_rate)
    
    # upsample metadata
    samples_per_metadata = convert(1)
    assert samples_per_metadata > 1

    # metadata samples
    values = np.array([target_value(d) for d in metadata_items], dtype=np.float32)
    samples = np.repeat(values, samples_per_metadata, axis=0)

    # slice samples matches time interval
    offset = convert(metadata_indexes[0])
    idx_start, idx_end = np.array([index_from_ts(ts, target_rate) for ts in interval]) - offset
    return samples[idx_start: idx_end]


####################################################################
# POINTING
####################################################################


def load_pointing_data(task_idx, path, chnl, task_rate, ts_origin, target_rate):

    """
    load and upsample pointing data for specific task

    Parameters
    ----------
    task_idx: int
        logical task number
    reader: digitalrf_wrapper.DigitalMetadataReader
        reader object for pointing data
    task_rate: float
        tasks per second
    ts_origin: float
        timestamp (sec) of first task
    sample_rate: target upsample rate

    Returns
    -------
    numpy.ndarray 
        (N,2) (float32) of upsampled (azimuth, elevation) tuples. Nan if chnl does not exist  
    """

    # time interval of this task 
    interval = [ts_from_index(idx, task_rate, ts_origin=ts_origin) for idx in [task_idx, task_idx+1]]

    try:
        # reader of pointing data
        reader = drf_wrapper.DigitalMetadataReader(path, chnl)
        print(path)
    except:
        # no reader - generate vector of NaN data
        N = int((interval[1] - interval[0]) * target_rate)
        return np.full((N, 2), np.nan)

    def target_value(item):
        return np.array([item['azimuth'], item['elevation']])


    # load pointing data as vector
    return load_metadata(reader, interval, target_rate, target_value)
