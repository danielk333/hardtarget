import numpy as np

####################################################################
# MOCK UP DATA
####################################################################

MOCK_DIM_1 = 135
MOCK_DIM_2 = 9600
SAMPLE_SIZE = 100000
INTEGRATION_SIZE = 100
RANGES_SIZE = 500
RANGE_RATE_SIZE = 600
ACCELERATIONS_SIZE = 500

mock_dim_1 = np.arange(MOCK_DIM_1)
mock_dim_2 = np.arange(MOCK_DIM_2)
sample_numbers = np.arange(SAMPLE_SIZE)
integration_index = np.arange(INTEGRATION_SIZE)
ranges = np.linspace(0, 1, RANGES_SIZE)
range_rates = np.linspace(0, 1, RANGE_RATE_SIZE)
accelerations = np.linspace(0, 1, ACCELERATIONS_SIZE)

range_rate_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
acceleration_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_zero_data = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
tx_power_data = np.random.rand(INTEGRATION_SIZE)
r_vec = np.random.rand(100)
v_vec = np.random.rand(100)
a_vec = np.random.rand(100)
g_vec = np.random.rand(100)
rgs = np.random.randint(0, 1000, size=(RANGES_SIZE), dtype=np.int32)
fvec = np.random.rand(RANGE_RATE_SIZE)
acceleration_phasors = np.random.rand(135, RANGE_RATE_SIZE) + 1j * np.random.rand(135, RANGE_RATE_SIZE)
rx_stencil = random_bool_array = np.random.choice([True, False], size=100000)
tx_stencil = random_bool_array = np.random.choice([True, False], size=100000)
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
        'rx_stencil': rx_stencil,
        'tx_stencil': tx_stencil,
        'rx_window_indices': rx_window_indices
    }
}