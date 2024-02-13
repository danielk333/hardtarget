import configparser
import numpy as np
import scipy.fft as fft
import scipy.constants
import pathlib

from hardtarget import drf_utils
from hardtarget.gmf import GMF_GRID_LIBS, GMF_OPTIMIZE_LIBS


####################################################################
# GET GMF PROCESSING PARAMS
#
# NOTE: all keys have to be lower-case here due to configparser
# NOTE: To add a new parameter that does not have a default value,
#       use None as the value
####################################################################

DEFAULT_PARAMS = {
    "gmf_grid_lib": ("c", str),
    "gmf_optimize_lib": ("c", str),
    "gmf_fine_tune": (False, bool),
    "node_gpus": (1, int),
    "n_ipp": (5, int),
    "ipp_offset": (0, int),
    "min_range_gate": (0, int),
    "max_range_gate": (-1, int),
    "range_gate_step": (1, int),
    "frequency_decimation": (1, int),
    "clutter_length": (0, int),
    "min_acceleration": (-200.0, float),
    "max_acceleration": (200.0, float),
    "acceleration_resolution": (0.2, float),
    "num_cohints_per_file": (100, int),
}

INT_PARAM_KEYS = [
    key
    for key, (_, tp) in DEFAULT_PARAMS.items()
    if tp is int
]

BOOL_PARAM_KEYS = [
    key
    for key, (_, tp) in DEFAULT_PARAMS.items()
    if tp is bool
]

FLOAT_PARAM_KEYS = [
    key
    for key, (_, tp) in DEFAULT_PARAMS.items()
    if tp is float
]


####################################################################
# LOAD GMF PROCESSING PARAMS FROM CONFIG
####################################################################

def load_gmf_processing_params(configfile=None):

    d = {
        key: val
        for key, (val, _) in DEFAULT_PARAMS.items()
    }

    assert configfile is not None, "Config file is NONE"
    cfg_pth = pathlib.Path(configfile)
    assert cfg_pth.exists(), f"Config file '{configfile}' does not exist"
    assert not cfg_pth.is_dir(), f"Config file '{configfile}' is a directory"

    if configfile is not None:
        config = configparser.ConfigParser()
        config.read(configfile)
        SECTION = "signal-processing"
        for key in config[SECTION].keys():
            if key not in DEFAULT_PARAMS:
                raise ValueError(f"'{key}' option found in config file not a valid config parameter")
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

    not_set_params = [key for key, val in d.items() if val is None]
    if len(not_set_params) > 0:
        raise ValueError(
            "Found config options without set values that does not have a default: \n"
            f"{not_set_params}"
        )
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

    # Use np.round and cast to int to avoid floating point errors in floor
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

    # TODO: document range gates properly, they are a bit of a mess right now
    # TODO: this current does not handle partial codes, add this functionality
    rgs_min = T_tx_start_samp
    rgs_max = T_rx_end_samp - tx_pulse_samps
    max_range_gate += rgs_max if max_range_gate < 0 else rgs_min
    min_range_gate += rgs_max if min_range_gate < 0 else rgs_min

    params_pro["range_gate_offset"] = rgs_min
    params_pro["rel_min_range_gate"] = min_range_gate - rgs_min
    params_pro["rel_max_range_gate"] = max_range_gate - rgs_min
    # reset range gates
    # range gates to search through
    # range gates are relative to tx start
    _rgs = np.arange(min_range_gate, max_range_gate, range_gate_step, dtype=np.int32)
    # total propagation range
    ranges = (_rgs - T_tx_start_samp) * scipy.constants.c / sample_rate  # m

    # make relative the stencil start
    rgs = _rgs - stencil_start_samp
    assert np.all(rgs > 0), "Computed range gates not compatible with stencils"

    params_der["rel_rgs"] = _rgs - rgs_min
    params_der["dec_rgs"] = np.floor(_rgs / frequency_decimation).astype(np.int32)
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
    assert tx_pulse_samps % frequency_decimation == 0, (
        "Pulse samples should be divisible by decimation to avoid edge effects\n"
        f"tx_pulse_samps / frequency_decimation = {tx_pulse_samps}/{frequency_decimation} ="
        f"{tx_pulse_samps / frequency_decimation}"
    )

    dec_sig_len = np.round(params_pro["read_length"] / frequency_decimation).astype(np.int64)
    dec_txlen = np.round(tx_pulse_samps / frequency_decimation).astype(np.int64)
    dec_ipp_samp = np.round(ipp_samp / frequency_decimation).astype(np.int64)
    dec_T_tx_start_samp = np.round(T_tx_start_samp / frequency_decimation).astype(np.int64)
    params_der["dec_signal_length"] = dec_sig_len

    # length of coherent integration
    params_pro["coh_int_samps"] = len(params_der["tx_stencil_indices"])
    params_pro["decimated_coh_int_samps"] = np.round(
        params_pro["coh_int_samps"] / frequency_decimation
    ).astype(np.int64)

    # Length of ffts
    # params_pro["n_fft"] = np.round(total_signal_length * sample_rate).astype(np.int64)
    params_pro["decimated_n_fft"] = dec_sig_len

    # frequency vector
    params_der["fvec"] = fft.fftfreq(
        params_pro["decimated_n_fft"],
        d=frequency_decimation / sample_rate,
    )  # Hz

    # range-rate is doppler-shift in hertz multiplied with wavelength
    wavelength = scipy.constants.c / (radar_frequency * 1e6)
    params_exp["wavelength"] = wavelength

    range_rates = wavelength * params_der["fvec"]
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
    dec_base_rx_window = np.arange(dec_txlen, dtype=np.int32) + dec_T_tx_start_samp
    dec_rx_window_blocks = [
        dec_base_rx_window + (ind + ipp_offset) * dec_ipp_samp
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
    rx_win_t_dec = rx_win_t[::frequency_decimation]

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

    max_phase_change = np.pi / wavelength * acc_interval * max_time2
    n_acc = np.ceil(max_phase_change/acceleration_resolution).astype(np.int64)
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
            -1j * np.pi / wavelength * acc * times2
        )

    params_pro["gmf_size"] = (params_pro["n_ranges"], )
    return params_exp, params_pro, params_der


####################################################################
# CONFIG HELPERS
####################################################################


def choose_gmf_implementation(params_pro):
    # gmf lib
    gmf_lib_name = params_pro.get("gmf_grid_lib", None)
    if gmf_lib_name is None:
        gmf_lib_name = "c" if "c" in GMF_GRID_LIBS else "numpy"
    elif gmf_lib_name not in GMF_GRID_LIBS:
        raise ValueError(
            f"Requested GMF gird lib '{gmf_lib_name}' not found in "
            "available libs, possible compilation errors in extensions\n"
            f"GMF_GRID_LIBS = {list(GMF_GRID_LIBS.keys())}"
        )

    # gmf optimize lib
    if params_pro.get("gmf_fine_tune", False):
        gmfo_lib_name = params_pro.get("gmf_optimize_lib", None)
        if gmfo_lib_name is None:
            gmfo_lib_name = "c" if "c" in GMF_OPTIMIZE_LIBS else "numpy"
        elif gmfo_lib_name not in GMF_OPTIMIZE_LIBS:
            raise ValueError(
                f"Requested GMF optimize lib '{gmfo_lib_name}' not found in "
                "available libs, possible compilation errors in extensions\n"
                f"GMF_OPTIMIZE_LIBS = {list(GMF_OPTIMIZE_LIBS.keys())}"
            )
    else:
        gmfo_lib_name = None

    return gmf_lib_name, gmfo_lib_name
