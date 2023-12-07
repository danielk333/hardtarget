# import pytest
import numpy as np
from hardtarget.gmf_out_utils import GMFOutArgs, dump_gmf_out, load_gmf_out
import tempfile
import pprint

####################################################################
# MOCK UP DATA
####################################################################

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
   
}

MOCK_DIM_1 = 135
MOCK_DIM_2 = 9600
SAMPLE_SIZE = 100000
RANGES_SIZE = 500
RANGE_RATE_SIZE = 600
ACCELERATIONS_SIZE = 500
INTEGRATION_SIZE = gmf_params["PRO"]["num_cohints_per_file"]

gmf_v_ind = np.random.randint(0, 1000, 
                              size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_a_ind = np.random.randint(0, 1000, 
                              size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_vals = np.random.randint(0, 1000, 
                             size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_dc = np.random.randint(0, 1000, 
                           size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
gmf_tx = np.random.rand(INTEGRATION_SIZE)
r_vec = np.random.rand(INTEGRATION_SIZE)
v_vec = np.random.rand(INTEGRATION_SIZE)
a_vec = np.random.rand(INTEGRATION_SIZE)
g_vec = np.random.rand(INTEGRATION_SIZE)
rgs = np.random.randint(0, 1000, size=(RANGES_SIZE), dtype=np.int32)
fvec = np.random.rand(RANGE_RATE_SIZE)
re = np.random.rand(MOCK_DIM_1, RANGE_RATE_SIZE)
im = 1j * np.random.rand(MOCK_DIM_1, RANGE_RATE_SIZE)
acceleration_phasors =  re + im
rx_stencil = np.random.choice([True, False], size=SAMPLE_SIZE)
tx_stencil = np.random.choice([True, False], size=SAMPLE_SIZE)
rx_window_indices = np.random.randint(0, 1000, size=(MOCK_DIM_2), 
                                      dtype=np.int32)
epoch_unix = 1.1


gmf_params['DER'] = {
    'rgs': rgs,
    'fvec': fvec,
    'acceleration_phasors': acceleration_phasors,
    'rx_stencil': rx_stencil,
    'tx_stencil': tx_stencil,
    'rx_window_indices': rx_window_indices
}

gmf_out_args = GMFOutArgs(
    num_cohints_per_file=gmf_params["PRO"]["num_cohints_per_file"],
    ranges=RANGES_SIZE,
    range_rates=RANGE_RATE_SIZE,
    accelerations=ACCELERATIONS_SIZE,
    sample_numbers=SAMPLE_SIZE,
    mock_dim_1=MOCK_DIM_1,
    mock_dim_2=MOCK_DIM_2,
    vals=gmf_vals,
    dc=gmf_dc,
    v_ind=gmf_v_ind,
    a_ind=gmf_a_ind,
    txp=gmf_tx,
    r_vec=r_vec,
    v_vec=v_vec,
    a_vec=a_vec,
    g_vec=g_vec,
    rgs=gmf_params["DER"]["rgs"],
    fvec=gmf_params["DER"]["fvec"],
    acceleration_phasors=gmf_params["DER"]["acceleration_phasors"],
    rx_stencil=gmf_params["DER"]["rx_stencil"],
    tx_stencil=gmf_params["DER"]["tx_stencil"],
    rx_window_indices=gmf_params["DER"]["rx_window_indices"],
    epoch=epoch_unix)

      
def test_write():

    # Create a temporary file for gmf out
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as temp_file:
        dump_gmf_out(gmf_out_args, gmf_params, temp_file.name)

        # Load gmf out
        items = load_gmf_out(temp_file.name)

        # Assert the result based on your expectations
        assert len(items) > 0

        pprint.pprint(items)

