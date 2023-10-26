import numpy as np
import scipy.constants

CONFIG_SECTIONS = [
    "radar-experiment",
    "signal-processing",
]

DEFAULT_PARAMS = {
    "n_ipp": 5,
    "sample_rate": 1000000,
    "n_range_gates": 10000,
    "range_gate_0": 200,
    "range_gate_step": 1,
    "frequency_decimation": 25,
    "ipp": 10000,
    "tx_pulse_length": 445,
    "ground_clutter_length": 1500,
    "min_acceleration": -400.0,
    "max_acceleration": 400.0,
    "acceleration_resolution": 0.2,
    "snr_thresh": 10.0,
    "doppler_sign": 1.0,
    "radar_frequency": 500e6,
    "round_trip_range": True,
    "num_cohints_per_file": 100,
}

INT_PARAM_KEYS = [
    "n_ipp",
    "n_range_gates",
    "range_gate_step",
    "range_gate_0",
    "frequency_decimation",
    "ipp",
    "tx_pulse_length",
    "ground_clutter_length",
    "num_cohints_per_file",
]

BOOL_PARAM_KEYS = ["round_trip_range"]

FLOAT_PARAM_KEYS = [
    "sample_rate",
    "min_acceleration",
    "max_acceleration",
    "acceleration_resolution",
    "snr_thresh",
    "doppler_sign",
    "radar_frequency",
]

DEFAULT_PARAM_KEYS = [key for key, _ in DEFAULT_PARAMS.items()]

DERIVED_PARAM_KEYS = [
    "rgs",
    "rgs_float",
    "ranges",
    "fvec",
    "n_fft",
    "fvec",
    "wavelength",
    "range_rates",
    "n_accelerations",
    "accs",
    "acc_phasors",
    "n_extra",
    "read_length",
    "rx_stencil",
    "tx_stencil",
]

REQUIRED_PARAM_KEYS = DEFAULT_PARAM_KEYS


def set_n_ranges(params):
    """
    Reset the number of range-gates in params. 

    Parameters
    ----------
    params: dict
        Dictionary with gmf parameters


    Notes
    -----
    This is useful when reanalyzing with better resolution
    to fine-tune the result
    """
    range_gate_0 = params["range_gate_0"]
    n_range_gates = params["n_range_gates"]
    range_gate_step = params["range_gate_step"]
    sample_rate = params["sample_rate"]

    # range gates to search through
    rgs = np.arange(n_range_gates) * range_gate_step + range_gate_0
    rgs_float = np.array(rgs, dtype=np.float32)

    # total propagation range
    ranges = rgs * scipy.constants.c / 1e3 / sample_rate

    params["rgs"] = rgs
    params["rgs_float"] = rgs_float
    params["ranges"] = ranges


####################################################################
# CHECK GMF PARAMS
####################################################################


def check_params(params):
    for key in REQUIRED_PARAM_KEYS:
        if key not in params:
            return False, f"missing parameter: {key}"
    return True, "ok"


####################################################################
# PROCESS GMF PARAMS
####################################################################


def process_params(params):
    """
    Updates given params dictionary in place, with computed values.

    Parameters
    ----------
    params: dict
        Dictionary with gmf parameters
    """

    # range gates
    set_n_ranges(params)

    n_ipp = params["n_ipp"]
    ipp = params["ipp"]
    frequency_decimation = params["frequency_decimation"]
    sample_rate = params["sample_rate"]
    radar_frequency = params["radar_frequency"]
    doppler_sign = params["doppler_sign"]
    max_acceleration = params["max_acceleration"]
    min_acceleration = params["min_acceleration"]
    acceleration_resolution = params["acceleration_resolution"]
    tx_pulse_length = params["tx_pulse_length"]
    ground_clutter_length = params["ground_clutter_length"]
    rgs = params["rgs"]

    # length of coherent integration
    params["n_fft"] = n_fft = n_ipp * ipp

    # frequency vector
    params["fvec"] = fvec = np.fft.fftfreq(
        int(n_fft / frequency_decimation),
        d=frequency_decimation / sample_rate,
    )

    # range-rate is doppler-shift in hertz multiplied with wavelength
    wavelength = scipy.constants.c / radar_frequency
    params["wavelength"] = wavelength

    range_rates = doppler_sign * wavelength * fvec
    params["range_rates"] = range_rates

    # Time vector
    times = frequency_decimation * np.arange(int(n_fft / frequency_decimation)) / sample_rate
    times2 = times**2.0

    # radar frequency in radians per second
    # om = 2.0 * np.pi * radar_frequency

    # these are the accelerations we'll try out
    tau = n_ipp * ipp / sample_rate

    # acceleration sampled with 0.2 radian steps at the end of the coherent integration window
    delta_a = max_acceleration - min_acceleration
    params["n_accelerations"] = n_accelerations = int(
        np.ceil(delta_a * (np.pi / wavelength) * tau**2.0 / acceleration_resolution)
    )

    params["accs"] = accs = np.linspace(
        min_acceleration, max_acceleration, num=n_accelerations
    )  # m/s**2

    params["acc_phasors"] = acc_phasors = np.zeros(
        [n_accelerations, int(n_fft / frequency_decimation)],
        dtype=np.complex64,
    )

    # precalculate phasors corresponding to different accelerations
    for ai, a in enumerate(accs):
        acc_phasors[ai, :] = np.exp(
            -1j * 2.0 * np.pi * (doppler_sign * 0.5 * accs[ai] / wavelength) * times2
        )

    # how many extra ipps do we need to read for coherent integration
    params["n_extra"] = n_extra = int(np.ceil(np.max(rgs) / ipp)) + 1

    # this stencil is used to block tx pulses and ground clutter
    params["read_length"] = read_length = n_fft + n_extra * ipp
    params["rx_stencil"] = rx_stencil = np.ones(read_length, dtype=np.float32)
    # this stencil is used to select tx pulses
    params["tx_stencil"] = tx_stencil = np.ones(read_length, dtype=np.float32)

    # for each coherently integrated IPP, create stencils
    for k in range(n_ipp + n_extra):
        tx_stencil[(k * ipp + tx_pulse_length): (k * ipp + ipp)] = 0.0
        # pad zeros to rx
        rx_stencil[(k * ipp): (k * ipp + tx_pulse_length + ground_clutter_length)] = 0.0
