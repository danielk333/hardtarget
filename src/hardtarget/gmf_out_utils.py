import numpy as np
import h5py
from collections import namedtuple
import datetime
import pathlib
import re

################################################################
# LOAD GMF OUT (H5)
#
# Parse groups and datasets recursively
################################################################


def collect_gmf_data(paths, mats=None, vecs=None):
    """Given a number of input h5 GMF paths, collect and concatenate some of the useful data.
    """
    data = None
    meta = None
    if mats is None:
        # Default mats
        mats = [
            "gmf",
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
            "gmf_peak",
            "tx_power",
            "t"
        ]
    derived = ["t", "nf_vec"]

    t_vec_pos = 0
    mats_data = {}
    vecs_data = {}
    for path in paths:
        with h5py.File(path, "r") as hf:
            for key in mats:
                if key in derived:
                    continue
                mats_data[key] = hf[key][()]
            for key in vecs:
                if key in derived:
                    continue
                vecs_data[key] = hf[key][()]

            if meta is None:
                meta = {
                    "ranges": hf["ranges"][()],
                    "range_rates": hf["range_rates"][()],
                    "accelerations": hf["accelerations"][()],
                    "range_gates": hf["processing"]["rgs"][()],
                }
                meta["processing"] = {key: val for key, val in hf["processing"].attrs.items()}
                meta["experiment"] = {key: val for key, val in hf["processing"].attrs.items()}
                for key, val in hf.attrs.items():
                    meta[key] = val

            _t_conv = (hf["processing"].attrs["n_ipp"][()] * hf["experiment"].attrs["ipp"][()]) * 1e-6

        # Additional useful parameters to calculate
        n_cohints = mats_data["gmf"].shape[0]
        vecs_data["t"] = np.arange(n_cohints) * _t_conv + t_vec_pos
        t_vec_pos = vecs_data["t"][-1] + 1
        nf_vec = np.nanmedian(mats_data["gmf_zero_frequency"], axis=0)
        nf_vec = nf_vec.reshape((1, nf_vec.size))
        mats_data["nf_vec"] = nf_vec

        if data is None:
            data = {}
            for key in mats:
                data[key] = mats_data[key]
            for key in vecs:
                data[key] = vecs_data[key]
        else:
            for key in mats:
                data[key] = np.append(data[key], mats_data[key], axis=0)
            for key in vecs:
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
    top = pathlib.Path(gmf_folder)
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

def dump_gmf_out(gmf_out_args, gmf_params, outfile):
    # TODO: it should be an option to dump a minimalist version of the gmf_out since
    #       it now has _a lot_ of metadata and parameters that might not be needed
    out = h5py.File(outfile, "w")

    # VARIABLES
    for key, item in define_variables(gmf_out_args).items():

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
        "rgs",
        "fvec",
        "decimated_sample_times",
        "acceleration_phasors",
        "rx_stencil",
        "tx_stencil",
        "rx_window_indices",
        "epoch"
    ],
)


####################################################################
# H5 VARIABLE DEFINITIONS
####################################################################

def define_variables(gmf_out_args):

    integration_index = np.arange(gmf_out_args.num_cohints_per_file, dtype=np.int64)
    rx_window_index = np.arange(len(gmf_out_args.rx_window_indices))

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
        "decimated_sample_times": {
            "data": gmf_out_args.decimated_sample_times,
            "long_name": "Time of decimated receiver samples in integration cycle",
            "units": "s",
            "scale": True
        },
        "rx_window_index": {
            "data": rx_window_index,
            "long_name": "Index within stenciled RX windows",
            "scale": True
        },
        "gmf": {
            "data": gmf_out_args.vals,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": "Generalized Matched Filter output values",
            # "group": "gmf"
        },
        "gmf_zero_frequency": {
            "data": gmf_out_args.dc,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": "Range dependant noise floor (0-frequency gmf output)",
            # "group": "gmf"
        },
        "range_rate_index": {
            "data": gmf_out_args.v_ind,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": (
                "If range_rate is reduced, contains the best range rate index "
                "for each left over axis"),
            # "group": "gmf"
        },
        "acceleration_index": {
            "data": gmf_out_args.a_ind,
            "dims": [("integration_index", "t"), ("ranges", "r")],
            "long_name": (
                "If acceleration is reduced, contains the best acceleration "
                "index for each left over axis"),
            # "group": "gmf"
        },
        "tx_power": {
            "data": gmf_out_args.txp,
            "dims": [("integration_index", "t")],
            "long_name": "Transmitted signal power",
            # "units": "W", # TODO: this is not converted to real power, just signal power
            # "group": "gmf"
        },
        "range_peak": {
            "data": gmf_out_args.r_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Range at peak GMF",
            # "group": "gmf"
        },
        "range_rate_peak": {
            "data": gmf_out_args.v_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Range rate at peak GMF",
            # "group": "gmf"
        },
        "acceleration_peak": {
            "data": gmf_out_args.a_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Acceleration at peak GMF",
            # "group": "gmf"
        },
        "gmf_peak": {
            "data": gmf_out_args.g_vec,
            "dims": [("integration_index", "t")],
            "long_name": "Peak GMF",
            # "group": "gmf"
        },
        "rgs": {
            "data": gmf_out_args.rgs,
            "dims": [("ranges", "r")],
            "long_name": "Range gates in signal samples",
            "group": "processing"
        },
        "fvec": {
            "data": gmf_out_args.fvec,
            "dims": [("range_rates", "v")],
            "long_name": "Searched doppler frequencies",
            "group": "processing",
            "units": "Hz"
        },
        "acceleration_phasors": {
            "data": gmf_out_args.acceleration_phasors,
            "dims": [("accelerations", "a"), ("decimated_sample_times", "t")],
            "long_name": (
                "Complex number representation of signal phase shift "
                "during decimated reception due to acceleration"
            ),
            "group": "processing",
        },
        "rx_stencil": {
            "data": gmf_out_args.rx_stencil,
            "dims": [("sample_numbers", "samples")],
            "long_name": "Stencil for selecting receiver samples in an integration cycle",
            "group": "processing"
        },
        "tx_stencil": {
            "data": gmf_out_args.tx_stencil,
            "dims": [("sample_numbers", "samples")],
            "long_name": "Stencil for selecting transmitter samples in an integration cycle",
            "group": "processing"
        },
        "rx_window_indices": {
            "data": gmf_out_args.rx_window_indices,
            "dims": [("rx_window_index", "rx_idx")],
            "long_name": (
                "Template receiver sample indices for selecting the length of a "
                "transmit signal within each radar cycle for an entire "
                "integration cycle, offset by range gate to select all signals "
                "from that range within each radar cycle"
            ),
            "group": "processing"
        }
    }
