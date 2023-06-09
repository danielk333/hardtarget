import configparser
import numpy as np
import scipy.constants

DEFAULT_PARAMS = {
    "n_ipp": 5,
    "sample_rate": 1000000,
    "n_range_gates": 10000,
    "range_gate_0": 200,
    "range_gate_step": 1,
    "frequency_decimation": 25,
    "ipp": 10000,
    "tx_pulse_length": 445,
    "tx_bit_length": 20,
    "ground_clutter_length": 1500,
    "min_acceleration": -400.0,
    "max_acceleration": 400.0,
    "acceleration_resolution": 0.2,
    "snr_thresh": 10.0,
    # "save_parameters": True,
    "doppler_sign": 1.0,
    "radar_frequency": 500e6,
    # "reanalyze": True,
    # "debug_plot": False,
    # "debug_plot_acc": False,
    # "debug_print": False,
    "round_trip_range": True,
    "num_cohints_per_file": 100,
    # "use_gpu": False,
    # "use_python": False,
    # "use_cpu": True,
}

INT_PARAMS = [
    'n_ipp', 
    'n_range_gates', 
    'range_gate_step',
    'range_gate_0', 
    'frequency_decimation',
    'ipp', 
    'tx_pulse_length',
    'ground_clutter_length',
    'num_cohints_per_file',
    't0'
]

BOOL_PARAMS = [
    # 'save_parameters',    
    # 'debug_plot',
    # 'debug_plot_acc',
    # 'debug_print',
    # 'use_gpu',
    # 'reanalyze', 
    'round_trip_range'
]

FLOAT_PARAMS = [
    'sample_rate', 
    'min_acceleration',
    'max_acceleration',
    'acceleration_resolution'
    'snr_thresh',
    'doppler_sign', 
    'radar_frequency' 
]



####################################################################
# LOAD GMF CONFIG
####################################################################

def load_gmf_config(config_file):
    """
    Load a gmf config file into to a dictionary
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    d = {}
    if 'config' in config:
        for key, value in config['config'].items():
            # Convert values to specific types
            if key in INT_PARAMS:
                d[key] = int(value)
            elif key in BOOL_PARAMS:
                d[key] = config.getboolean('config', key)
            elif key in ['rx_channel', 'tx_channel', 'output_dir']:
                d[key] = value.strip('"')       
            elif key in FLOAT_PARAMS:
                d[key] = float(value)
            else:                
                pass
    return d



def set_n_ranges(params):
    """
    Reset the number of range-gates. This is useful when reanalyzing with better resolution
    to fine-tune the result
    """
    range_gate_0 = params["range_gate_0"]
    n_range_gates = params["n_range_gates"]
    range_gate_step = params["range_gate_step"]
    sample_rate = params["sample_rate"]

    # range gates to search through
    rgs = (
        np.arange(n_range_gates) * range_gate_step + range_gate_0
    )
    rgs_float = np.array(rgs, dtype=np.float32)

    # total propagation range
    ranges = rgs * scipy.constants.c / 1e3 / sample_rate

    params["range_gate_0"] = range_gate_0
    params["n_range_gates"] = n_range_gates
    params["rgs"] = rgs
    params["rgs_float"] = rgs_float
    params["ranges"] = ranges



####################################################################
# PROCESS GMF PARAMS
####################################################################

def process_gmf_params(params):

    n_ipp = params["n_ipp"]
    ipp = params["ipp"]
    frequency_decimation = params["frequency_decimation"]
    sample_rate = params["sample_rate"]
    radar_frequency = params["radar_frequency"]
    doppler_sign = params["doppler_sign"]
    max_acceleration = params["max_acceleration"]
    min_acceleration = params["min_acceleration"]
    acceleration_resolution = params["acceleration_resolution"]
    rgs = params["rgs"]
    tx_pulse_length = params["tx_pulse_length"]
    ground_clutter_length = params["ground_clutter_length"]

    # length of coherent integration
    params["n_fft"] = n_fft = n_ipp * ipp

    # frequency vector
    params["fvec"] = fvec = np.fft.fftfreq(
        int(n_fft / frequency_decimation),
        d=frequency_decimation / sample_rate,
    )

    # range gates
    set_n_ranges(params)

    # range-rate is doppler-shift in hertz multiplied with wavelength
    params["wavelength"] = wavelength = scipy.constants.c / radar_frequency
    params["range_rates"] = range_rates = doppler_sign * wavelength * fvec

    # Time vector
    times = (
        frequency_decimation
        * np.arange(int(n_fft / frequency_decimation))
        / sample_rate
    )
    times2 = times**2.0

    # radar frequency in radians per second
    om = 2.0 * np.pi * radar_frequency

    # these are the accelerations we'll try out
    tau = n_ipp * ipp / sample_rate

    # acceleration sampled with 0.2 radian steps at the end of the coherent integration window
    delta_a = max_acceleration - min_acceleration
    params["n_accelerations"] = n_accelerations = int(
        np.ceil(
            delta_a
            * (np.pi / wavelength)
            * tau**2.0
            / acceleration_resolution
        )
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
            -1j
            * 2.0
            * np.pi
            * (doppler_sign * 0.5 * accs[ai] / wavelength)
            * times2
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
        tx_stencil[
            (k * ipp + tx_pulse_length) : (k * ipp + ipp)
        ] = 0.0
        # pad zeros to rx
        rx_stencil[
            (k * ipp) : (
                k * ipp + tx_pulse_length + ground_clutter_length
            )
        ] = 0.0





####################################################################
# LOAD GMF OPTONS
####################################################################

def load_gmf_params(config_file):
    d = {}.update(DEFAULT_params, load_gmf_config(config_file))
    return d





