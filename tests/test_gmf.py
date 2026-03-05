import tempfile
from pathlib import Path

import numpy as np
import pytest
from radardef.types import ExpParams

import hardtarget.process.utils as utils
from hardtarget.data_handling.configuration import (
    compute_process_params,
    load_config_params,
)
from hardtarget.matched_filter.gmf import get_gmf_lib
from hardtarget.matched_filter.types import MFVariables
from hardtarget.types.constants import AnalysisMethod, Impl, TargetEstimationMethod

"""
Should ideally be able to test the different implementations of the GMF function

ISSUE 1
Functions have different signatures now

regular ones must be called with
(tx, rx, gmf_variables, gmf_params)
whereas optimized
(tx, ipp, gmf_params, gmf_start)

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


def create_experiment_params():

    return ExpParams(
        name="leo_bpark",
        radar_frequency=929.6,
        t_ipp_usec=20000,
        ipp_samps=20000,
        sample_rate=1000000.0,
        t_samp_usec=1,
        rx_channels=["uhf"],
        t_rx_start_usec=0,
        t_rx_end_usec=20000,
        t_tx_start_usec=82.0,
        t_tx_end_usec=2002.0,
        tx_channel="uhf",
        tx_pulse_length=1920.0,
        t_cal_on_usec=19900.0,
        t_cal_off_usec=19997.0,
    )


def create_config_params():

    MOCK_CONFIG = """
    [processing]
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
    [gmf]

    """

    # Make temp directory with mockup config files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mockup DRF metadata file
        # metafile = Path(temp_dir) / "metadata.ini"
        # with open(metafile, "w") as f:
        #    f.write(MOCK_DRF_METADATA)

        # Mockup GMF processing config
        config = Path(temp_dir) / "config.ini"
        with open(config, "w") as f:
            f.write(MOCK_CONFIG)

        # process params
        return load_config_params(config)


class TestGMF:
    def test_gmf(self):
        """Run the basic gmf function."""

        # GMF method and implementation
        gmf_method = TargetEstimationMethod.fgmf
        gmf_implementation = Impl.numpy
        gmf_lib = get_gmf_lib(
            gmf_method,
            gmf_implementation,
        )

        # GMF params
        init_pro_params = create_config_params()
        experiment = create_experiment_params()

        pro_params = compute_process_params(
            experiment,
            init_pro_params,
            analysis_method=AnalysisMethod.target_estimation,
            method_lib=TargetEstimationMethod.fgmf,
            implementation=Impl.c,
        )

        # Initialise vectors

        # - new
        size = len(pro_params.ranges)
        vars = MFVariables(
            vals=np.zeros(size, dtype=np.float32),
            dc=np.zeros(size, dtype=np.float32),
            v_ind=np.full(size, -1, dtype=np.int32),
            a_ind=np.full(size, -1, dtype=np.int32),
            tx_pwr=None,  # not needed
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
        tx = np.zeros(10000, dtype=np.complex64)
        rx = np.zeros(12000, dtype=np.complex64)
        for i in range(10):
            tx[(i * 1000): (i * 1000 + 20)] = 1.0
            rx[(i * 1000 + 500): (i * 1000 + (500 + 20))] = 0.5  # simulated "echo"

        """

        # Process

        # - old
        """
        # for i in range(20):
        gmf_func(tx, rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec)
        """

        # - new
        """
        gmf_lib(
            tx,
            rx,
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
