import numpy as np
import h5py
from collections import namedtuple


################################################################
# LOAD GMF OUT (H5)
#
# Parse groups and datasets recursively
################################################################


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
    return items


def inspect_h5_leaf(obj, path):
    item = {}

    if isinstance(obj, h5py.Dataset):
        item["scale"] = is_scale(obj)
        if obj.dtype == "object":
            item["type"] = "string"
            item["dtype"] = obj.dtype
            item["value"] = obj[()].decode('utf-8')
        elif obj.shape == ():
            item["type"] = "scalar"
            item["dtype"] = obj.dtype
            item["value"] = obj[()]
        else:
            item["type"] = "dataset"
            item["shape"] = obj.shape
            item["dtype"] = obj.dtype
            item["value"] = obj
    else:
        item["type"] = "other"
        item["value"]: obj

    return path, item


def load_gmf_out(file):
    with h5py.File(file, "r") as f:
        return inspect_h5_node(f)


####################################################################
# DUMP GMF OUT
####################################################################

def dump_gmf_out(gmf_out_args, gmf_params, outfile):

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
        exp_grp[key] = val

    # GMF PROCESSING PARAMS
    if "processing" not in out:
        out.create_group("processing")
    pro_grp = out["processing"]
    for key, val in gmf_params["PRO"].items():
        pro_grp[key] = val

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
        # r_ind,
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
# DROPPED - As long as there is no reduction in range dimension
# "range_index": {
#     "data": r_ind,
#     "dims": [("integration_index", "t"), ("ranges", "r")],
#     "long_name": "If range is reduced, contains the best range index for each left over axis",
#     "group": "gmf"
# }

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
