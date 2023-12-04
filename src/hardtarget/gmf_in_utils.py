import configparser
import numpy as np
import scipy.constants
import pathlib
from hardtarget import drf_utils
from collections import namedtuple


####################################################################
# GMF INPUT DATA
####################################################################

"""Container for compacting the variables set by the GMF function."""
GMFVariables = namedtuple(
    "GMFVariables",
    [
        "vals",  # match function values reduced over the requested axis
        "dc",  # 0-frequency gmf output
        "r_ind",  # best fitting range
        "v_ind",  # best fitting range-rate
        "a_ind",  # best fitting range-rate change
    ],
)


####################################################################
# GET GMF PROCESSING PARAMS
####################################################################

DEFAULT_PARAMS = {
    "gmflib": "c",
    "node_GPUs": 1,
    "n_ipp": 5,
    "ipp_offset": 0,
    "min_range_gate": 0,
    "max_range_gate": -1,
    "range_gate_step": 1,
    "frequency_decimation": 2,
    "clutter_length": 1500,
    "min_acceleration": -400.0,
    "max_acceleration": 400.0,
    "acceleration_resolution": 0.2,
    "doppler_sign": 1.0,
    "round_trip_range": True,
    "num_cohints_per_file": 100,
    "reduce_range": False,
    "reduce_range_rate": True,
    "reduce_acceleration": True,
}

INT_PARAM_KEYS = [
    "node_GPUs",
    "n_ipp",
    "ipp_offset",
    "min_range_gate",
    "max_range_gate",
    "range_gate_step",
    "frequency_decimation",
    "clutter_length",
    "num_cohints_per_file",
]

BOOL_PARAM_KEYS = [
    "round_trip_range",
    "reduce_range",
    "reduce_range_rate",
    "reduce_acceleration",
]

FLOAT_PARAM_KEYS = [
    "min_acceleration",
    "max_acceleration",
    "acceleration_resolution",
    "doppler_sign",
]


####################################################################
# LOAD GMF PROCESSING PARAMS FROM CONFIG
####################################################################

def load_gmf_processing_params(configfile=None):

    d = {**DEFAULT_PARAMS}

    assert configfile is not None, "Config file is NONE"
    cfg_pth = pathlib.Path(configfile)
    assert cfg_pth.exists(), f"Config file '{configfile}' does not exist"
    assert not cfg_pth.is_dir(), f"Config file '{configfile}' is a directory"

    if configfile is not None:
        config = configparser.ConfigParser()
        config.read(configfile)
        SECTION = "signal-processing"
        for key in config[SECTION].keys():
            # Convert values to specific types
            if key in INT_PARAM_KEYS:
                d[key] = config.getint(SECTION, key)
            elif key in BOOL_PARAM_KEYS:
                d[key] = config.getboolean(SECTION, key)
            elif key in FLOAT_PARAM_KEYS:
                d[key] = config.getfloat(SECTION, key)
            else:
                # string
                d[key] = config.get(SECTION, key).strip("'").strip('"')
    
    # GMF library implementation currently implicitly assumes reduction over
    # range_rate and acceleration.
    assert d["reduce_range"] is False, "Not supported: reduce_range: True"
    assert d["reduce_range_rate"] is True, "Not supported: reduce_range_rate: False"
    assert d["reduce_acceleration"] is True, "Not supported: reduce_acceleration: False"

    return d


####################################################################
# LOAD ALL GMF PARAMS
####################################################################


def load_gmf_params(drf_srcdir, gmf_configfile):
    """
        initalise params from
        - experiment (hardtarget_drf)
        - signal processing config (gmf_config)
        calculate additional derived params
    """

    print(drf_srcdir)

    # drf experiment parmeters
    params_exp = drf_utils.load_hardtarget_drf_params(drf_srcdir)
    # gmf processing params
    params_pro = load_gmf_processing_params(gmf_configfile)
    # gmf derived params
    derived_params = compute_derived_gmf_params(params_exp, params_pro) 

    return {
        "EXP": params_exp,
        "PRO": params_pro,
        "DER": derived_params
    }


####################################################################
# COMPUTE DERIVED GMF PARAMS
####################################################################

def compute_derived_gmf_params(params_exp, params_pro):

    # experiment params
    sample_rate = params_exp["sample_rate"]
    ipp = params_exp["ipp"]
    radar_frequency = params_exp["radar_frequency"]
    doppler_sign = params_exp["doppler_sign"]
    rx_start = params_exp["rx_start"]
    rx_end = params_exp["rx_end"]
    tx_start = params_exp["tx_start"]
    tx_end = params_exp["tx_end"]

    # gmf processing params
    min_range_gate = params_pro["min_range_gate"]
    max_range_gate = params_pro["max_range_gate"]
    range_gate_step = params_pro["range_gate_step"]
    n_ipp = params_pro["n_ipp"]
    ipp_offset = params_pro["ipp_offset"]
    frequency_decimation = params_pro["frequency_decimation"]
    max_acceleration = params_pro["max_acceleration"]
    min_acceleration = params_pro["min_acceleration"]
    acceleration_resolution = params_pro["acceleration_resolution"]
    clutter_length = params_pro["clutter_length"]

    # compute derived params
    p = {}

    ipp_samp = np.round(ipp * 1e-6 * sample_rate).astype(np.int64)
    p["ipp_samp"] = ipp_samp

    # Use np.round and case to int to avoid floating point errors in floor
    T_rx_start_samp = np.round(rx_start * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(rx_end * 1e-6 * sample_rate).astype(np.int64)
    T_tx_start_samp = np.round(tx_start * 1e-6 * sample_rate).astype(np.int64)
    T_tx_end_samp = np.round(tx_end * 1e-6 * sample_rate).astype(np.int64)
    tx_pulse_length = params_exp.get("tx_pulse_length", None)
    if tx_pulse_length is not None:
        tx_pulse_samps = np.round(tx_pulse_length * 1e-6 * sample_rate).astype(np.int64)
        _assert_msg = "TX pulse lengths does not correspond to tx start and stop values"
        assert tx_pulse_samps == T_tx_end_samp - T_tx_start_samp, _assert_msg
    else:
        tx_pulse_samps = T_tx_end_samp - T_tx_start_samp
    p["tx_pulse_samps"] = tx_pulse_samps

    # Relevant rx samples
    stencil_start_samp = T_rx_start_samp if T_rx_start_samp > T_tx_end_samp else T_tx_end_samp
    stencil_start_samp += clutter_length

    p["n_rx_samples"] = T_rx_end_samp - stencil_start_samp
    p["stencil_start_samp"] = stencil_start_samp

    # TODO: this current does not handle partial codes, add this functionality
    rgs_min = T_tx_start_samp
    rgs_max = T_rx_end_samp - tx_pulse_samps
    max_range_gate += rgs_max if max_range_gate < 0 else rgs_min
    min_range_gate += rgs_max if min_range_gate < 0 else rgs_min
    # reset range gates
    # range gates to search through
    # range gates are relative to tx start
    rgs = np.arange(min_range_gate, max_range_gate, range_gate_step, dtype=np.int32)
    # total propagation range
    ranges = (rgs - T_tx_start_samp) * scipy.constants.c / sample_rate  # m

    # make relative the stencil start
    rgs -= stencil_start_samp
    p["rgs"] = rgs
    p["ranges"] = ranges
    p["n_ranges"] = len(ranges)

    # how many extra ipps do we need to read for coherent integration
    p["n_extra"] = ipp_offset

    # length of coherent integration
    p["n_fft"] = n_ipp * tx_pulse_samps
    p["decimated_n_fft"] = int(p["n_fft"] / frequency_decimation)

    # frequency vector
    p["fvec"] = fvec = np.fft.fftfreq(
        p["decimated_n_fft"],
        d=frequency_decimation / sample_rate,
    )

    # range-rate is doppler-shift in hertz multiplied with wavelength
    wavelength = scipy.constants.c / (radar_frequency * 1e6)
    p["wavelength"] = wavelength

    range_rates = doppler_sign * wavelength * fvec
    p["range_rates"] = range_rates  # m/s
    p["n_range_rates"] = len(range_rates)

    # cyclic range gate selector
    # TODO: this can be generalized for a-periodic codes ect
    base_rx_window = np.arange(p["tx_pulse_samps"], dtype=np.int32)
    rx_window_blocks = [
        base_rx_window + ind * p["n_rx_samples"]
        for ind in range(n_ipp)
    ]
    rx_window_indices = np.concatenate(rx_window_blocks)
    # TODO: This is part of future generalization to partial codes too
    # p["rx_window_blocks"] = rx_window_blocks
    p["rx_window_indices"] = rx_window_indices

    # We are modelling the target at the time of the first tx-pulse scattered
    # of the target, the first sample of the coherent integration
    # is a "good enough" timestamp but its actually that
    # + (range_gate + stencil_start_samp)/2 for monostatic

    # calculate midpoint of decimated vectors
    rx_win_dec = np.mean(rx_window_indices.copy().reshape(-1, frequency_decimation), axis=-1)

    # Time vector relative detected range-gate
    times = rx_win_dec / sample_rate
    times2 = times**2.0

    # acceleration sampled with steps at the end of the coherent integration window
    # acceleration_resolution is in radians!
    max_time2 = np.max(times2)
    acc_interval = max_acceleration - min_acceleration
    max_phase_change = 2.0 * np.pi * (0.5 * acc_interval / wavelength) * max_time2
    p["n_accelerations"] = int(np.ceil(max_phase_change/acceleration_resolution))
    if p["n_accelerations"] == 0:
        p["n_accelerations"] = 1

    p["accelerations"] = np.linspace(
        min_acceleration, max_acceleration, num=p["n_accelerations"]
    )  # m/s^2

    p["acceleration_phasors"] = np.zeros(
        [p["n_accelerations"], p["decimated_n_fft"]],
        dtype=np.complex64,
    )

    # precalculate phasors corresponding to different accelerations
    for ai, a in enumerate(p["accelerations"]):
        p["acceleration_phasors"][ai, :] = np.exp(
            -1j * 2.0 * np.pi * (doppler_sign * 0.5 * p["accelerations"][ai] / wavelength) * times2
        )

    # Read length to include all pulses to be searched
    p["read_length"] = (n_ipp + ipp_offset) * ipp_samp

    # this stencil is used to block tx pulses and ground clutter
    p["rx_stencil"] = np.full((p["read_length"],), False, dtype=bool)
    # this stencil is used to select tx pulses
    p["tx_stencil"] = np.full((p["read_length"],), False, dtype=bool)

    # for each coherently integrated IPP, create stencils
    for k in range(n_ipp):
        rx0 = ((k + ipp_offset) * ipp_samp + stencil_start_samp)
        rx1 = ((k + ipp_offset) * ipp_samp + T_rx_end_samp)
        p["rx_stencil"][rx0:rx1] = True

        tx0 = (k * ipp_samp + T_tx_start_samp)
        tx1 = (k * ipp_samp + T_tx_end_samp)
        p["tx_stencil"][tx0:tx1] = True

    reduce_axis = [
        params_pro["reduce_range"],
        params_pro["reduce_range_rate"],
        params_pro["reduce_acceleration"],
    ]
    gmf_size = [
        p["n_ranges"],
        p["n_range_rates"],
        p["n_accelerations"],
    ]
    gmf_size = [s for red, s in zip(reduce_axis, gmf_size) if not red]

    p["gmf_size"] = gmf_size
    p["reduce_axis"] = reduce_axis

    return p
