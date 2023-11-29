import h5py
import numpy as np

####################################################################
# MOCK UP DATA
####################################################################

INTEGRATION_SIZE = 100
RANGES_SIZE = 500
RANGE_RATE_SIZE = 600
ACCELERATIONS_SIZE = 500

integration_index = np.arange(INTEGRATION_SIZE)
ranges = np.linspace(0, 1, RANGES_SIZE)
range_rates = np.linspace(0, 1, RANGE_RATE_SIZE)
accelerations = np.linspace(0, 1, ACCELERATIONS_SIZE)
range_rate_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
acceleration_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_zero_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
tx_power_data = np.random.rand(INTEGRATION_SIZE)
rgs = np.random.randint(0, 1000, size=(RANGES_SIZE), dtype=np.int32)
fvec = np.random.rand(RANGE_RATE_SIZE)
acceleration_phasors = np.random.rand(135, RANGE_RATE_SIZE) + 1j * np.random.rand(135, RANGE_RATE_SIZE)
rx_stensil = random_bool_array = np.random.choice([True, False], size=100000)
tx_stensil = random_bool_array = np.random.choice([True, False], size=100000)
rx_window_indices = np.random.randint(0, 1000, size=(9600), dtype=np.int32)


gmf_params = {
    'EXP': {
        'cal_off': 19997.0,
        'cal_on': 19900.0,
        'doppler_sign': -1.0,
        'file_secs': 12.8,
        'ipp': 20000,
        'name': 'leo_bpark',
        'radar_frequency': 929.6,
        'round_trip_range': False,
        'rx_channel': 'uhf',
        'rx_end': 19997.0,
        'rx_start': 2.0,
        'sample_rate': 1000000,
        'tx_channel': 'uhf',
        'tx_end': 2002.0,
        'tx_pulse_length': 1920.0,
        'tx_start': 82.0,
        'version': '2.1u'},
    'PRO': {
        'acceleration_resolution': 0.2,
        'clutter_length': 1500,
        'doppler_sign': -1.0,
        'frequency_decimation': 16,
        'gmflib': 'c',
        'ipp_offset': 0,
        'max_acceleration': 300.0,
        'max_range_gate': 7000,
        'min_acceleration': -300.0,
        'min_range_gate': 6500,
        'n_ipp': 5,
        'node_GPUs': 1,
        'node_gpus': '1',
        'num_cohints_per_file': 100,
        'range_gate_step': 1,
        'reduce_acceleration': True,
        'reduce_range': False,
        'reduce_range_rate': True,
        'round_trip_range': True
    },
    'DER': {
        'rgs': rgs,
        'fvec': fvec,
        'acceleration_phasors': acceleration_phasors,
        'rx_stencil': rx_stensil,
        'tx_stencil': tx_stensil,
        'rx_window_indices': rx_window_indices
    }
}


####################################################################
# H5 DIMENSIONS AND VARIABLES 
####################################################################


VARIABLE_MAP = {
    "integration_index": {
        "data": integration_index,
        "long_name": "Integration index within this file relative the epoch"
    },
    "ranges": {
        "data": ranges,
        "long_name": "Matched filter ranges",
        "units": "m"
    },
    "range_rates": {
        "data": range_rates,
        "long_name": "Matched filter range rates",
        "units": "m/s"
    },
    "accelerations": {
        "data": accelerations,
        "long_name": "Matched filter range accelerations",
        "units": "m/s^2"
    },
    "gmf": {
        "data": gmf_data,
        "dims": [("integration_index", "t"), ("ranges", "r")],
        "long_name": "Generalized Matched Filter output values",
    },
    "gmf_zero_frequency": {
        "data": gmf_zero_data,
        "dims": [("integration_index", "t"), ("ranges", "r")],
        "long_name": "Range dependant noise floor (0-frequency gmf output)"
    },
    # DROPPED - As long as there is no reduction in range dimension
    # "range_index": {
    #     "data": range_data,
    #     "dims": [("integration_index", "t"), ("ranges", "r")],
    #     "long_name": "If range is reduced, contains the best range index for each left over axis"
    # }
    "range_rate_index": {
        "data": range_rate_data,
        "dims": [("integration_index", "t"), ("ranges", "r")],
        "long_name": "If range_rate is reduced, contains the best range rate index for each left over axis",
    },
    "acceleration_index": {
        "data": acceleration_data,
        "dims": [("integration_index", "t"), ("ranges", "r")],
        "long_name": "If acceleration is reduced, contains the best acceleration index for each left over axis",
    },
    "tx_power": {
        "data": tx_power_data, 
        "dims": [("integration_index", "t")],
        "long_name": "Measured transmitted power",
        "units": "W"
    }
}


if __name__ == "__main__":

    out = h5py.File('example.h5', "w")

    # EXPERIMENT PARAMS
    exp_grp = out.create_group("experiment")
    for key, val in gmf_params["EXP"].items():
        exp_grp[key] = val

    # GMF PROCESSING PARAMS
    pro_grp = out.create_group("processing")
    for key, val in gmf_params["PRO"].items():
        pro_grp[key] = val

    # VECTORS
    VECTOR_PARAMS = [
        "rgs",
        "fvec",
        "acceleration_phasors",
        "rx_stencil",
        "tx_stencil",
        "rx_window_indices",
    ]
    vector_grp = out.create_group("vector_params")
    for key in VECTOR_PARAMS:
        vector_grp[key] = gmf_params["DER"][key]

    # EPOCH
    out["epoch_unix"] = 1    

    # VARIABLES
    var_grp = out.create_group("gmf")
    for key, item in VARIABLE_MAP.items():
        ds = var_grp.create_dataset(key, data=item["data"])
        if "dims" not in item:
            ds.make_scale(key)
        else:
            for idx, (scale_key, label) in enumerate(item["dims"]):
                scale = var_grp[scale_key]
                ds.dims[idx].attach_scale(scale)
                ds.dims[idx].label = label
        ds.attrs["long_name"] = item["long_name"]
        if "units" in item:
            ds.attrs["units"] = item["units"]


    out.close()
