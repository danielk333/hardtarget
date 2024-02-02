import configparser
import numpy as np
import scipy.fft as fft
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
        "dc",  # 0-frequency gmf output as a function of range
        "r_ind",  # best fitting range
        "v_ind",  # best fitting range-rate
        "a_ind",  # best fitting range-rate change
        "peak",  # fine-tuned peak in range, range-rate & range-rate change
        "peak_val",  # value of fine-tuned peak
    ],
)


####################################################################
# GET GMF PROCESSING PARAMS
#
# NOTE: all keys have to be lower-case here due to configparser
####################################################################

DEFAULT_PARAMS = {
    "gmf_grid_lib": "c",
    "gmf_optimize_lib": "c",
    "gmf_fine_tune": True,
    "node_gpus": 1,
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
    "node_gpus",
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
    """Loads and calculates parameters needed to run analysis.

    First load params from
     - experiment (hardtarget_drf)
     - signal processing config (gmf_config)
    Then calculate additional derived parameters that are implementation specific
    and cumbersome, such as signal stencils and acceleration phasors.

    Output format is structured as:
        {
            "EXP": experiment parameters,
            "PRO": signal processing parameters,
            "DER": derived parameters for internal use,
        }

    All parameters are assumed to be constant after initialization.
    """

    # drf experiment parmeters
    params_exp = drf_utils.load_hardtarget_drf_params(drf_srcdir)
    # gmf processing params
    params_pro = load_gmf_processing_params(gmf_configfile)
    # gmf derived params
    params_exp, params_pro, params_der = compute_derived_gmf_params(params_exp, params_pro)

    return {
        "EXP": params_exp,
        "PRO": params_pro,
        "DER": params_der
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
    params_der = {}

    ipp_samp = np.round(ipp * 1e-6 * sample_rate).astype(np.int64)
    params_exp["ipp_samp"] = ipp_samp

    # Use np.round and case to int to avoid floating point errors in floor
    T_rx_start_samp = np.round(rx_start * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(rx_end * 1e-6 * sample_rate).astype(np.int64)
    T_tx_start_samp = np.round(tx_start * 1e-6 * sample_rate).astype(np.int64)
    T_tx_end_samp = np.round(tx_end * 1e-6 * sample_rate).astype(np.int64)
    tx_pulse_length = params_exp.get("tx_pulse_length", None)
    if tx_pulse_length is not None:
        tx_pulse_samps = np.round(tx_pulse_length * 1e-6 * sample_rate).astype(np.int64)
        assert tx_pulse_samps == T_tx_end_samp - T_tx_start_samp, (
            "TX pulse lengths does not correspond to tx start and stop values"
        )
    else:
        tx_pulse_samps = T_tx_end_samp - T_tx_start_samp
        params_exp["tx_pulse_length"] = 1e6 * (T_tx_end_samp - T_tx_start_samp) / sample_rate
    params_exp["tx_pulse_samps"] = tx_pulse_samps

    # Relevant rx samples
    stencil_start_samp = T_rx_start_samp if T_rx_start_samp > T_tx_end_samp else T_tx_end_samp
    stencil_start_samp += clutter_length

    params_der["n_rx_samples"] = T_rx_end_samp - stencil_start_samp
    params_der["stencil_start_samp"] = stencil_start_samp

    # TODO: this current does not handle partial codes, add this functionality
    rgs_min = T_tx_start_samp
    rgs_max = T_rx_end_samp - tx_pulse_samps
    max_range_gate += rgs_max if max_range_gate < 0 else rgs_min
    min_range_gate += rgs_max if min_range_gate < 0 else rgs_min

    params_pro["min_range_gate"] = min_range_gate
    params_pro["max_range_gate"] = max_range_gate
    # reset range gates
    # range gates to search through
    # range gates are relative to tx start
    rgs = np.arange(min_range_gate, max_range_gate, range_gate_step, dtype=np.int32)
    # total propagation range
    # TODO - assumes config round_trip_range is True
    ranges = (rgs - T_tx_start_samp) * scipy.constants.c / sample_rate  # m

    # make relative the stencil start
    rgs -= stencil_start_samp
    params_der["rgs"] = rgs
    params_der["ranges"] = ranges
    params_pro["n_ranges"] = len(ranges)

    # how many extra ipps do we need to read for coherent integration
    params_pro["n_extra"] = ipp_offset

    # Read length to include all pulses to be searched
    params_pro["read_length"] = (n_ipp + ipp_offset) * ipp_samp

    # this stencil is used to block tx pulses and ground clutter
    params_der["rx_stencil"] = np.full((params_pro["read_length"],), False, dtype=bool)
    # this stencil is used to select tx pulses
    params_der["tx_stencil"] = np.full((params_pro["read_length"],), False, dtype=bool)

    # for each coherently integrated IPP, create stencils
    for k in range(n_ipp):
        rx0 = ((k + ipp_offset) * ipp_samp + stencil_start_samp)
        rx1 = ((k + ipp_offset) * ipp_samp + T_rx_end_samp)
        params_der["rx_stencil"][rx0:rx1] = True

        tx0 = (k * ipp_samp + T_tx_start_samp)
        tx1 = (k * ipp_samp + T_tx_end_samp)
        params_der["tx_stencil"][tx0:tx1] = True

    params_der["rx_stencil_indices"] = np.argwhere(params_der["rx_stencil"]).flatten()
    params_der["tx_stencil_indices"] = np.argwhere(params_der["tx_stencil"]).flatten()

    # Decimated signals
    dec_sample_rate = sample_rate // frequency_decimation
    sig_int = ipp * 1e-6 * (n_ipp + ipp_offset)
    dec_sig_len = int(sig_int * dec_sample_rate)
    dec_txlen = tx_pulse_samps // frequency_decimation
    dec_ipp_samp = ipp_samp // frequency_decimation
    params_der["dec_signal_length"] = dec_sig_len

    # length of coherent integration
    params_pro["coh_int_samps"] = n_ipp * tx_pulse_samps
    params_pro["decimated_coh_int_samps"] = int(params_pro["coh_int_samps"] / frequency_decimation)

    # Length of ffts
    params_pro["n_fft"] = int(sig_int * sample_rate)
    params_pro["decimated_n_fft"] = dec_sig_len

    # frequency vector
    params_der["fvec"] = fvec = fft.fftfreq(
        params_pro["decimated_n_fft"],
        d=frequency_decimation / sample_rate,
    )  # Hz

    # range-rate is doppler-shift in hertz multiplied with wavelength
    wavelength = scipy.constants.c / (radar_frequency * 1e6)
    params_exp["wavelength"] = wavelength

    range_rates = doppler_sign * wavelength * fvec
    params_der["range_rates"] = range_rates  # m/s
    params_pro["n_range_rates"] = len(range_rates)

    # cyclic range gate selector - in the index space of stenciled RX signals
    # TODO: this can be generalized for a-periodic codes ect
    base_rx_window = np.arange(params_exp["tx_pulse_samps"], dtype=np.int32)
    rx_window_blocks = [
        base_rx_window + ind * params_der["n_rx_samples"]
        for ind in range(n_ipp)
    ]
    rx_window_indices = np.concatenate(rx_window_blocks)
    # TODO: This is part of future generalization to partial codes too
    # params_der["rx_window_blocks"] = rx_window_blocks
    params_der["rx_window_indices"] = rx_window_indices

    # The following calculates the corresponding windows in the decimated signal
    # for use for constructing the vector that FFT can operate on
    #  - in the index ipp cycle, not of stenciled RX signals
    dec_base_rx_window = np.arange(dec_txlen, dtype=np.int32)
    dec_rx_window_blocks = [
        dec_base_rx_window + ind * dec_ipp_samp
        for ind in range(n_ipp)
    ]
    dec_rx_window_indices = np.concatenate(dec_rx_window_blocks)
    params_der["dec_rx_window_indices"] = dec_rx_window_indices

    # TODO: check all index and time assumptions, if one was wrong the more can be...
    # TODO: the sample gates can actually move as a function of velocity + acceleration
    # so maybe the rx_window_indices should also change so we dont miss-align samples?

    # We are modelling the target parameters at the time of the first cycle sample
    # calculate midpoint of decimated vectors (going to highest level inds: cycle samples)
    rx_win_t = params_der["rx_stencil_indices"][rx_window_indices] / sample_rate
    rx_win_t_dec = np.mean(rx_win_t.reshape(-1, frequency_decimation), axis=-1)

    # Time vector relative detected range-gate
    times2 = rx_win_t_dec**2.0
    params_der["decimated_sample_times"] = rx_win_t_dec
    params_der["sample_times"] = rx_win_t

    # acceleration sampled with steps at the end of the coherent integration window
    # acceleration_resolution is in radians!
    max_time2 = np.max(times2)
    acc_interval = max_acceleration - min_acceleration

    # TODO: maybe not? for now make sure we have a 0 acceleration
    assert min_acceleration <= 0 and max_acceleration >= 0, "acceleration needs to cover 0"

    max_phase_change = 2.0 * np.pi * (0.5 * acc_interval / wavelength) * max_time2
    n_acc = int(np.ceil(max_phase_change/acceleration_resolution))
    d_acc = acc_interval / n_acc

    params_der["accelerations"] = np.concatenate([
        np.arange(min_acceleration, 0, d_acc, dtype=np.float64),
        np.arange(0, max_acceleration, d_acc, dtype=np.float64),
    ])  # m/s^2
    params_pro["n_accelerations"] = len(params_der["accelerations"])

    params_der["acceleration_phasors"] = np.zeros(
        [params_pro["n_accelerations"], params_pro["decimated_coh_int_samps"]],
        dtype=np.complex64,
    )

    # precalculate phasors corresponding to different accelerations
    for ai, acc in enumerate(params_der["accelerations"]):
        params_der["acceleration_phasors"][ai, :] = np.exp(
            -1j * 2.0 * np.pi * (doppler_sign * 0.5 * acc / wavelength) * times2
        )

    reduce_axis = [
        params_pro["reduce_range"],
        params_pro["reduce_range_rate"],
        params_pro["reduce_acceleration"],
    ]
    gmf_size = [
        params_pro["n_ranges"],
        params_pro["n_range_rates"],
        params_pro["n_accelerations"],
    ]
    gmf_size = [s for red, s in zip(reduce_axis, gmf_size) if not red]

    params_pro["gmf_size"] = gmf_size
    params_pro["reduce_axis"] = reduce_axis

    return params_exp, params_pro, params_der
