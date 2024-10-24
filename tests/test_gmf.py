import pytest
import numpy as np
import hardtarget.analysis.utils as utils
from hardtarget.gmf import get_estimation_method
from hardtarget.configuration import load_gmf_params
import tempfile
from pathlib import Path

"""
Should ideally be able to test the different implementations of the GMF function

ISSUE 1
Functions have different signatures now

regular ones must be called with
(z_tx, z_rx, gmf_variables, gmf_params)
whereas optimized
(z_tx, z_ipp, gmf_params, gmf_start)

ISSUE 2
implementations of GMF functions depend on gmf_params,
gmf_params = {"DER": {}, "PRO": {}}
This way of organizing parameters is not motivated in the context of 
gmf_function implementation.

ISSUE 3
Implementation of GMF functions use different subsets of parameters from gmf_params

ISSUE 4
gmf_variables are defined in hardtaget.analyis.utils. If they are to be part of the
gmf signature, they should be defined within the gmf module.

ISSUE 5
gmf_variables (as defined in hardtarget.analysis.util) include one attribute tx_pow,
which is not needed by gmf functions.

"""


def create_gmf_params():
    """
    Create mockup gmf params for testing gmf
    """

    MOCK_DRF_METADATA = """
    [Experiment]
        name = leo_bpark
        version = 2.1u
        sample_rate = 1000000
        ipp = 20000
        file_secs = 12.8
        tx_pulse_length = 1920.0
        rx_channel = uhf
        rx_start = 0
        rx_end = 20000
        tx_channel = uhf
        tx_start = 82.0
        tx_end = 2002.0
        cal_on = 19900.0
        cal_off = 19997.0
        radar_frequency = 929.6

    [Bounds]
        ts_start = 1445511612.8
        ts_end = 1445551228.8
        start = 2015-10-22T11:00:12.800000
        end = 2015-10-22T22:00:28.800000
    """

    MOCK_GMF_CONFIG = """
    [signal-processing]
        n_ipp=10
        ipp_offset=0
        min_range_gate=6800
        max_range_gate=7280
        # min_range_gate=3420
        # max_range_gate=10000
        min_acceleration=-300.0
        max_acceleration=300.0
        range_gate_step=1
        frequency_decimation=16
        num_cohints_per_file=10
        node_gpus=1
        dpt_ipp_delay_parameter=5
    """

    # Make temp directory with mockup config files
    with tempfile.TemporaryDirectory() as temp_dir:

        # Mockup DRF metadata file
        metafile = Path(temp_dir) / "metadata.ini"
        with open(metafile, "w") as f:
            f.write(MOCK_DRF_METADATA)

        # Mockup GMF processing config
        config = Path(temp_dir) / "config.ini"
        with open(config, "w") as f:
            f.write(MOCK_GMF_CONFIG)

        # GMF params
        return load_gmf_params(temp_dir, str(config))





class TestGMF:


    def test_gmf(self):
        """Run the basic gmf function."""

        # GMF method and implementation
        gmf_method = "fgmf"
        gmf_implementation = "numpy"
        gmf_lib, gmf_libtype = get_estimation_method(
            gmf_implementation,
            gmf_method
        )

        # GMF params
        gmf_params = create_gmf_params()

        # Initialise vectors
        
        # - new
        size = gmf_params["PRO"]["n_ranges"]
        gmf_vars = utils.GMFVariables(
            vals = np.zeros(size, dtype=np.float32),
            dc = np.zeros(size, dtype=np.float32),
            v_ind = np.full(size, -1, dtype=np.int32),
            a_ind = np.full(size, -1, dtype=np.int32),
            tx_pwr = None  # not needed
        )

        # - old
        """        
        dec = 10
        acc_phasors = np.zeros([20, 1000], dtype=np.complex64)
        acc_phasors[0, :] = 1.0
        rgs = np.arange(1000, dtype=np.float32)
        
        n_r = len(rgs)
        gmf_vec = np.zeros(n_r, dtype=np.float32)
        gmf_dc_vec = np.zeros(n_r, dtype=np.float32)
        v_vec = np.zeros(n_r, dtype=np.float32)
        a_vec = np.zeros(n_r, dtype=np.float32)
        """

        # Mockup signal
       
        # - old       
        """
        z_tx = np.zeros(10000, dtype=np.complex64)
        z_rx = np.zeros(12000, dtype=np.complex64)
        for i in range(10):
            z_tx[(i * 1000): (i * 1000 + 20)] = 1.0
            z_rx[(i * 1000 + 500): (i * 1000 + (500 + 20))] = 0.5  # simulated "echo"
                    
        """

        # Process

        # - old
        """        
        # for i in range(20):
        gmf_func(z_tx, z_rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec)
        """

        # - new
        """
        gmf_lib(
            z_tx,
            z_rx,
            gmf_vars,
            gmf_params
        )
        """

        # Expectation

        # - old
        """
        expected = {"ri": 500, "gmf_vec": 1e04, "v_vec": 0.0, "a_vec": 0.0}
        """

        # Evaluate expectation

        # - old
        """
        ri = np.argmax(gmf_vec)
        assert expected["ri"] == ri
        assert expected["gmf_vec"] == gmf_vec[ri]
        assert expected["v_vec"] == v_vec[ri]
        assert expected["a_vec"] == a_vec[ri]
        """

        assert True
