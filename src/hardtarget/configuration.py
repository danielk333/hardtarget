"""

Configuration
=============

Main module for configuring the analysis of the input data and calculating the
relevant parameters needed for analysis.

The analysis always assumes the `digital_rf` (drf) radar data consists of a
continuous stream of signal samples. If in reality there was no samples taken
during an interval the sample stream should be zero-padded during this time.
As such the signal sample stream is split up into regular cycles. Each cycle
occurs back-to-back and the time a cycle takes is called an inter-pulse-period,
or IPP.

Every analysis consists of a received signal (RX) and a transmitted signal (TX).
These signals can either be superimposed in the same channel or be in different
channels in the drf structure. Either way, there are several segments within
each cycle that we need to extract and index. Hence, there are three main levels
of signal indices within one cycle, which we will call Index Levels (IL's):

 0) Signal samples (can be same channel)
    - RX signal (size=IPP length)
    - TX signal (size=IPP length)
 0d) Decimated signal samples
    - RX signal (size=IPP length / decimation)
 1) Stenciled samples:
    - RX window (size=chosen reception length, i.e. all range gates)
    - TX pulse (size=length of transmitted pulse)
 2) Target range-gate:
    - RX pulse (size=length of transmitted pulse, offset by chosen range gate)

The IL-0d is a bit of a special case as temporal decimation (piece-wise sums of
neighbors) only maintains desirable properties (both statistical and signal
vise) if it is done on an isochronal and continuous sample stream. As such,
decimation operation only occurs on a level 0 signal.

Illustration of the above:

    IL-0     : |0123456789...................................|
    Signal   : |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|
    IL-0d    : | 0  1  2  3  4  5  6  7  8  9  .  .  .  .  . |
    Decimated: | x  x  x  x  x  x  x  x  x  x  x  x  x  x  x |
    IL-1 tx  : |  012345                                     |
    TX pulse : |--xxxxxx-------------------------------------|
    IL-1 rx  : |                0123456789..............     |
    RX window: |----------------xxxxxxxxxxxxxxxxxxxxxxxx-----|
    IL-2     : |                           012345            |
    RX pulse : |---------------------------xxxxxx------------|

In the above example, the first sample of the TX pulse would have index 2 in
terms of IL-0 but would have index 0 in terms of IL-1. The first sample of IL-0d
would represent samples 0, 1, and 2 in IL-0. Similarly the first sample of the
RX pulse would have index 27 in terms of IL-0, index 11 in IL-1 and index 0 in
IL-2. Also, index 0, 1, and 2 in the RX pulse IL-2 would be associated with only
index 9 in IL-0d.

The general strategy for decimation is to zero-pad the target vector if the
decimation does not evenly divide the vector.

In the analysis, several of these cycles are stacked on top of each other. This
means that when selecting the IL-1 tx samples the resulting array is no longer
isochronal and continues but instead has a jump in the middle, e.g.

    IL-1 tx  : |  012345        6789..        ......      |
    TX pulse : |--xxxxxx--------xxxxxx--------xxxxxx------|

Since range from a transmitter converted to time is measured from the start of
transmission (i.e. the time between when the leading edge of the wave leaves the
transmitter and reaches the receiver), the range gates are also always measured
in samples relative the start of the TX pulse + 1, in terms of IL-0. Which means
that sample of range-gate 0 has traveled the time of 1 sample. The largest range
gate is at the end of reception, which basically means only one sample of the TX
could be measured.

# TODO: decimation runs on JUST the top 1 level of direct signal

# TODO: the current analysis might not handle partial codes, add this functionality

"""

import configparser
import numpy as np
import scipy.fft as fft
import scipy.constants
import pathlib

from hardtarget import drf_utils
from hardtarget.gmf import get_default_method


class ConfigError(Exception):
    pass


DEFAULT_IMPL, DEFAULT_METHOD = get_default_method()

####################################################################
# GET GMF PROCESSING PARAMS
#
# NOTE: all keys have to be lower-case here due to configparser
# NOTE: To add a new parameter that does not have a default value,
#       use None as the value
####################################################################

DEFAULT_PARAMS = {
    "gmf_implementation": (DEFAULT_IMPL, str),
    "gmf_method": (DEFAULT_METHOD, str),
    "node_gpus": (1, int),
    "n_ipp": (10, int),
    "ipp_offset": (0, int),
    "dpt_ipp_delay_parameter": (5, int),
    "min_range_gate": (0, int),
    "max_range_gate": (-1, int),
    "range_gate_step": (1, int),
    "frequency_decimation": (1, int),
    "clutter_length": (0, int),
    "min_acceleration": (-200.0, float),
    "max_acceleration": (200.0, float),
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
                raise ConfigError(f"'{key}' option found in config file not a valid config parameter")
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
        raise ConfigError(
            "The following parameters are required and do not have defaults "
            "but were not found in the config file: \n"
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
    rel_min_range_gate = params_pro["min_range_gate"]
    rel_max_range_gate = params_pro["max_range_gate"]
    range_gate_step = params_pro["range_gate_step"]
    n_ipp = params_pro["n_ipp"]
    ipp_offset = params_pro["ipp_offset"]
    frequency_decimation = params_pro["frequency_decimation"]
    max_acceleration = params_pro["max_acceleration"]
    min_acceleration = params_pro["min_acceleration"]
    tau_ipp = params_pro["dpt_ipp_delay_parameter"]

    # compute derived params
    params_der = {}

    ########################################
    # Experiment and instrument parameters
    ########################################

    ipp_samp = np.round(ipp * 1e-6 * sample_rate).astype(np.int64)
    params_exp["ipp_samp"] = ipp_samp

    # how many extra ipps do we need to read for coherent integration
    params_pro["n_extra"] = ipp_offset

    # Use np.round and cast to int to avoid floating point errors in floor
    T_rx_start_samp = np.round(rx_start * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(rx_end * 1e-6 * sample_rate).astype(np.int64)
    T_tx_start_samp = np.round(tx_start * 1e-6 * sample_rate).astype(np.int64)
    T_tx_end_samp = np.round(tx_end * 1e-6 * sample_rate).astype(np.int64)
    params_exp["T_rx_start_samp"] = T_rx_start_samp
    params_exp["T_rx_end_samp"] = T_rx_end_samp
    params_exp["T_tx_start_samp"] = T_tx_start_samp
    params_exp["T_tx_end_samp"] = T_tx_end_samp

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

    # Read length to include all pulses to be searched
    params_pro["read_length"] = (n_ipp + ipp_offset) * ipp_samp
    params_pro["decimated_read_length"] = np.ceil(
        params_pro["read_length"] / frequency_decimation
    ).astype(np.int64)

    # range-rate is doppler-shift in hertz multiplied with wavelength
    wavelength = scipy.constants.c / (radar_frequency * 1e6)
    params_exp["wavelength"] = wavelength

    ########################################
    # Range related parameters
    ########################################

    # range gates are relative to tx start + 1
    il0_rgs_min = T_tx_start_samp + 1
    # TODO: this is one of the parts of not handling partial codes
    il0_rgs_max = T_rx_end_samp - tx_pulse_samps
    il0_max_range_gate = rel_max_range_gate
    il0_max_range_gate += il0_rgs_min if rel_max_range_gate >= 0 else il0_rgs_max
    il0_min_range_gate = rel_min_range_gate
    il0_min_range_gate += il0_rgs_min if rel_min_range_gate >= 0 else il0_rgs_max

    assert il0_max_range_gate <= T_rx_end_samp - tx_pulse_samps, (
        "start range gate cannot be after than RX end"
    )
    assert il0_max_range_gate > T_rx_start_samp, (
        "start range gate cannot be before than RX start"
    )
    assert il0_min_range_gate <= T_rx_end_samp - tx_pulse_samps, (
        "end range gate cannot be after than RX end"
    )
    assert il0_min_range_gate > T_rx_start_samp, (
        "end range gate cannot be before than RX start"
    )

    min_range_gate = il0_min_range_gate - il0_rgs_min
    max_range_gate = il0_max_range_gate - il0_rgs_min

    params_pro["il0_min_range_gate"] = il0_min_range_gate
    params_pro["il0_max_range_gate"] = il0_max_range_gate
    params_pro["min_range_gate"] = min_range_gate
    params_pro["max_range_gate"] = max_range_gate

    rgs = np.arange(min_range_gate, max_range_gate, range_gate_step, dtype=np.int32)
    il0_rgs = np.arange(il0_min_range_gate, il0_max_range_gate, range_gate_step, dtype=np.int32)
    rel_rgs = rgs - min_range_gate
    # total propagation range
    ranges = (rgs + 1) * scipy.constants.c / sample_rate  # m
    assert np.all(rgs > 0), "Computed range gates not compatible with stencils"

    params_der["rgs"] = rgs
    params_der["il0_rgs"] = il0_rgs
    params_der["rel_rgs"] = rel_rgs
    params_der["ranges"] = ranges
    params_pro["n_ranges"] = len(ranges)

    ########################################
    # Signal indexing levels
    ########################################

    # this stencil is used to block tx pulses and ground clutter
    params_der["rx_stencil"] = np.full((params_pro["read_length"],), False, dtype=bool)
    # this stencil is used to select tx pulses
    params_der["tx_stencil"] = np.full((params_pro["read_length"],), False, dtype=bool)

    # for each coherently integrated IPP, create stencils
    for k in range(n_ipp):
        # start of pulse within range-gates, thus include also the entire pulse at the end
        rx0 = ((k + ipp_offset) * ipp_samp + il0_min_range_gate)
        rx1 = ((k + ipp_offset) * ipp_samp + il0_max_range_gate + tx_pulse_samps)
        params_der["rx_stencil"][rx0:rx1] = True

        tx0 = (k * ipp_samp + T_tx_start_samp)
        tx1 = (k * ipp_samp + T_tx_end_samp)
        params_der["tx_stencil"][tx0:tx1] = True

    params_der["il0_rx_stencil_indices"] = np.argwhere(params_der["rx_stencil"]).flatten()
    params_der["il0_tx_stencil_indices"] = np.argwhere(params_der["tx_stencil"]).flatten()

    # length of coherent integration
    params_pro["coh_int_samps"] = len(params_der["il0_tx_stencil_indices"])

    # Decimated signals
    # TODO: this can probably be allowed if we truncate/pad the end or start of the decimated vector
    assert tx_pulse_samps % frequency_decimation == 0, (
        "Pulse samples should be divisible by decimation to avoid edge effects\n"
        f"tx_pulse_samps / frequency_decimation = {tx_pulse_samps}/{frequency_decimation} ="
        f"{tx_pulse_samps / frequency_decimation}"
    )
    # TODO: this can be avoided by padding the stencil and having a second 0-stencil
    # TODO: better assert messages if keep
    assert params_pro["n_ranges"] % frequency_decimation == 0, (
        "range-gate interval not compatible with decimation: "
        f"{params_pro['n_ranges']} % {frequency_decimation} = "
        f"{params_pro['n_ranges'] % frequency_decimation}"
    )
    assert ipp_samp % frequency_decimation == 0, (
        "ipp-samples length not compatible with decimation: "
        f"{ipp_samp} % {frequency_decimation} = "
        f"{ipp_samp % frequency_decimation}"
    )
    assert params_pro["coh_int_samps"] % frequency_decimation == 0, (
        "range-gate interval not compatible with decimation: "
        f"{params_pro['coh_int_samps']} % {frequency_decimation} = "
        f"{params_pro['coh_int_samps'] % frequency_decimation}"
    )

    # cyclic range gate selector - in the index space of stenciled RX signals
    # TODO: this can be generalized for a-periodic codes ect
    base_rx_window = np.arange(params_exp["tx_pulse_samps"], dtype=np.int32)
    d_window = params_exp["tx_pulse_samps"] + params_pro["n_ranges"]
    il1_rx_window_blocks = [
        base_rx_window + ind * d_window
        for ind in range(n_ipp)
    ]
    il1_rx_window_indices = np.concatenate(il1_rx_window_blocks)
    params_der["il1_rx_window_indices"] = il1_rx_window_indices
    il0_rx_window_indices = params_der["il0_rx_stencil_indices"][il1_rx_window_indices]
    params_der["il0_rx_window_indices"] = il0_rx_window_indices

    il0_dec_rx_window_indices = il0_rx_window_indices[::frequency_decimation] // frequency_decimation
    params_der["il0_dec_rx_window_indices"] = il0_dec_rx_window_indices

    ########################################
    # Velocity related parameters
    ########################################

    # frequency vector
    params_der["fft_frequencies"] = fft.fftfreq(
        params_pro["decimated_read_length"],
        d=frequency_decimation / sample_rate,
    )  # Hz

    range_rates = wavelength * params_der["fft_frequencies"]
    params_der["range_rates"] = range_rates  # m/s
    params_pro["n_range_rates"] = len(range_rates)

    ########################################
    # Acceleration related parameters
    # TODO: at what time are we exactly modelling the acceleration
    ########################################

    # Sample times in the decimated il0d vector
    rx_win_t = np.arange(params_pro["read_length"]) / sample_rate
    rx_win_t = rx_win_t[params_der["il0_rx_stencil_indices"]]
    rx_win_t = rx_win_t[params_der["il1_rx_window_indices"]]
    # take the decimated times as the center-points of decimated samples
    rx_win_t_dec = np.mean(rx_win_t.reshape(-1, frequency_decimation), axis=-1)

    # TODO: the sample gates can actually move as a function of velocity + acceleration
    #       so maybe the rx_window_indices should also change so we dont miss-align samples?

    # Time vector relative detected range-gate
    times2 = rx_win_t_dec**2.0
    params_der["decimated_sample_times"] = rx_win_t_dec
    params_der["sample_times"] = rx_win_t

    # tau_ipp
    dec_tau_samp = np.floor(ipp_samp * tau_ipp / frequency_decimation).astype(np.int64)
    step = 2 * dec_tau_samp * frequency_decimation / sample_rate
    max_accels_len = params_pro["decimated_read_length"] - dec_tau_samp
    dpt_accels = fft.fftshift(fft.fftfreq(max_accels_len, d=frequency_decimation/sample_rate))
    dpt_accels = dpt_accels * wavelength * 2 / step
    accel_resolution = wavelength * 2 * sample_rate / (max_accels_len * frequency_decimation * step)

    params_pro["dec_tau_samp"] = dec_tau_samp

    # filter accels
    accel_inds = np.logical_and(dpt_accels >= min_acceleration, dpt_accels <= max_acceleration)
    fgmf_accels = dpt_accels[accel_inds]
    params_der["inds_accelerations"] = np.argwhere(accel_inds).flatten()
    params_der["accelerations"] = dpt_accels  # m/s^2
    params_der["acceleration_step"] = accel_resolution
    params_der["fgmf_accelerations"] = fgmf_accels  # m/s^2
    params_pro["n_accelerations"] = len(dpt_accels)

    # precalculate phasors corresponding to different accelerations
    phasors = np.exp(
        -1j * np.pi * params_der["accelerations"][:, None] * times2[None, :] / wavelength
    ).astype(np.complex64)
    params_der["acceleration_phasors"] = phasors
    params_der["fgmf_acceleration_phasors"] = phasors[accel_inds, :].copy()

    return params_exp, params_pro, params_der
