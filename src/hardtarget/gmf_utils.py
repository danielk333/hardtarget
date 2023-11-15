import configparser
import numpy as np
import scipy.constants
import pathlib
from hardtarget import drf_utils


####################################################################
# GET GMF PROCESSING PARAMS
####################################################################

DEFAULT_PARAMS = {
    "gmflib": "c",
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

AXIS_PARAM_KEYS = {
    "ranges": ("Matched filter ranges", "m"),
    "range_rates": ("Matched filter range rates", "m/s"),
    "accelerations": ("Matched filter range accelerations", "m/s^2"),
}
VECTOR_PARAM_KEYS = [
    "rgs",
    "fvec",
    "acceleration_phasors",
    "rx_stencil",
    "tx_stencil",
    "rx_window_indices",
]


####################################################################
# LOAD GMF PROCESSING PARAMS FROM CONFIG
####################################################################

def load_gmf_processing_params(configfile=None):

    d = {**DEFAULT_PARAMS}

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

    params = {
        **drf_utils.load_hardtarget_drf_params(drf_srcdir),
        **load_gmf_processing_params(gmf_configfile)
    }

    # variable access
    min_range_gate = params["min_range_gate"]
    max_range_gate = params["max_range_gate"]
    range_gate_step = params["range_gate_step"]
    sample_rate = params["sample_rate"]
    n_ipp = params["n_ipp"]
    ipp = params["ipp"]
    ipp_offset = params["ipp_offset"]
    frequency_decimation = params["frequency_decimation"]
    radar_frequency = params["radar_frequency"]
    doppler_sign = params["doppler_sign"]
    max_acceleration = params["max_acceleration"]
    min_acceleration = params["min_acceleration"]
    acceleration_resolution = params["acceleration_resolution"]
    clutter_length = params["clutter_length"]

    ipp_samp = np.round(ipp * 1e-6 * sample_rate).astype(np.int64)
    params["ipp_samp"] = ipp_samp

    # Use np.round and case to int to avoid floating point errors in floor
    T_rx_start_samp = np.round(params["rx_start"] * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(params["rx_end"] * 1e-6 * sample_rate).astype(np.int64)
    T_tx_start_samp = np.round(params["tx_start"] * 1e-6 * sample_rate).astype(np.int64)
    T_tx_end_samp = np.round(params["tx_end"] * 1e-6 * sample_rate).astype(np.int64)
    if "tx_pulse_length" in params:
        tx_pulse_samps = np.round(params["tx_pulse_length"] * 1e-6 * sample_rate).astype(np.int64)
        _assert_msg = "TX pulse lengths does not correspond to tx start and stop values"
        assert tx_pulse_samps == T_tx_end_samp - T_tx_start_samp, _assert_msg
    else:
        tx_pulse_samps = T_tx_end_samp - T_tx_start_samp
    params["tx_pulse_samps"] = tx_pulse_samps

    # Relevant rx samples
    rx_start = T_rx_start_samp if T_rx_start_samp > T_tx_end_samp else T_tx_end_samp
    rx_start += clutter_length

    params["n_rx_samples"] = T_rx_end_samp - rx_start

    # TODO: this current does not handle partial codes, add this functionality
    rgs_min = T_tx_start_samp
    rgs_max = T_rx_end_samp - tx_pulse_samps
    max_range_gate += rgs_max if max_range_gate < 0 else rgs_min
    min_range_gate += rgs_max if min_range_gate < 0 else rgs_min
    # reset range gates
    # range gates to search through
    # range gates are relative to tx start
    rgs = np.arange(min_range_gate, max_range_gate, range_gate_step)
    # total propagation range
    ranges = (rgs - T_tx_start_samp) * scipy.constants.c / sample_rate  # m

    # make relative the stencil start
    rgs -= rx_start
    params["rgs"] = rgs
    params["ranges"] = ranges
    params["n_ranges"] = len(ranges)

    # how many extra ipps do we need to read for coherent integration
    params["n_extra"] = ipp_offset

    # length of coherent integration
    params["n_fft"] = n_ipp * tx_pulse_samps
    params["decimated_n_fft"] = int(params["n_fft"] / frequency_decimation)

    # frequency vector
    params["fvec"] = fvec = np.fft.fftfreq(
        params["decimated_n_fft"],
        d=frequency_decimation / sample_rate,
    )

    # range-rate is doppler-shift in hertz multiplied with wavelength
    wavelength = scipy.constants.c / (radar_frequency * 1e6)
    params["wavelength"] = wavelength

    range_rates = doppler_sign * wavelength * fvec
    params["range_rates"] = range_rates  # m/s
    params["n_range_rates"] = len(range_rates)

    # cyclic range gate selector
    # TODO: this can be generalized for a-periodic codes ect
    base_rx_window = np.arange(params["tx_pulse_samps"], dtype=np.int32)
    rx_window_indices = np.concatenate([
        base_rx_window + ind * params["n_rx_samples"]
        for ind in range(params["n_ipp"])
    ])
    params["rx_window_indices"] = rx_window_indices

    # calculate midpoint of decimated vectors
    rx_win_dec = np.mean(rx_window_indices.copy().reshape(-1, frequency_decimation), axis=-1)
    # Time vector
    times = rx_win_dec / sample_rate
    times2 = times**2.0

    # these are the accelerations we'll try out
    tau = n_ipp * ipp * 1e-6

    # acceleration sampled with steps at the end of the coherent integration window
    # acceleration_resolution is in radians!
    delta_a = max_acceleration - min_acceleration
    params["n_accelerations"] = int(
        np.ceil(delta_a * (np.pi / wavelength) * tau**2.0 / acceleration_resolution)
    )
    if params["n_accelerations"] == 0:
        params["n_accelerations"] = 1

    params["accelerations"] = np.linspace(
        min_acceleration, max_acceleration, num=params["n_accelerations"]
    )  # m/s^2

    params["acceleration_phasors"] = np.zeros(
        [params["n_accelerations"], params["decimated_n_fft"]],
        dtype=np.complex64,
    )

    # precalculate phasors corresponding to different accelerations
    for ai, a in enumerate(params["accelerations"]):
        params["acceleration_phasors"][ai, :] = np.exp(
            -1j * 2.0 * np.pi * (doppler_sign * 0.5 * params["accelerations"][ai] / wavelength) * times2
        )

    # Read length to include all pulses to be searched
    params["read_length"] = (n_ipp + ipp_offset) * ipp_samp

    # this stencil is used to block tx pulses and ground clutter
    params["rx_stencil"] = np.full((params["read_length"],), False, dtype=bool)
    # this stencil is used to select tx pulses
    params["tx_stencil"] = np.full((params["read_length"],), False, dtype=bool)

    # for each coherently integrated IPP, create stencils
    for k in range(n_ipp):
        rx0 = ((k + ipp_offset) * ipp_samp + rx_start)
        rx1 = ((k + ipp_offset) * ipp_samp + T_rx_end_samp)
        params["rx_stencil"][rx0:rx1] = True

        tx0 = (k * ipp_samp + T_tx_start_samp)
        tx1 = (k * ipp_samp + T_tx_end_samp)
        params["tx_stencil"][tx0:tx1] = True

    return params
