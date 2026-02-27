import tempfile
from pathlib import Path

import numpy as np

from hardtarget.data_handling.configuration import compute_process_params
from hardtarget.data_handling.store_params import dump_params_to_file
from hardtarget.matched_filter.gmf.types import GMFCfgParams
from hardtarget.matched_filter.types import MFOutArgs
from hardtarget.plotting.load_data import load_analysed_data
from hardtarget.process import GMFProcess
from hardtarget.types.constants import AnalysisMethod, Impl, TargetEstimationMethod
from hardtarget.types.types import ExpParams


def test_store_and_load():
    exp_org = ExpParams(
        name="leo_bpark",
        radar_frequency=929.6,
        t_ipp_usec=20000,
        ipp_samps=20000,
        t_samp_usec=1,
        rx_channels=["uhf"],
        tx_channel="uhf",
        tx_pulse_length=1920,
        t_rx_start_usec=2.0,
        t_rx_end_usec=19997.0,
        t_tx_start_usec=82.0,
        t_tx_end_usec=2002.0,
        t_cal_on_usec=19900.0,
        t_cal_off_usec=19997.0,
        wavelength=1,
        sample_rate=1000000,
        data=None,
        code=None,
        pulse=None,
    )

    cfg_org = GMFCfgParams(
        node_gpus=1,
        n_ipp=5,
        ipp_offset=0,
        min_range_gate=6500,
        max_range_gate=7000,
        range_gate_step=1,
        frequency_decimation=10,
        clutter_length=1500,
        min_acceleration=-300.0,
        max_acceleration=300.0,
        num_cohints_per_file=100,
    )

    MOCK_DIM_1: int = 135
    MOCK_DIM_2: int = 9600
    SAMPLE_SIZE: int = 100000
    RANGES_SIZE: int = 500
    RANGE_RATE_SIZE: int = 600
    ACCELERATIONS_SIZE: int = 500
    INTEGRATION_SIZE: int = cfg_org.num_cohints_per_file

    gmf_v_ind = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
    gmf_a_ind = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
    gmf_vals = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
    gmf_dc = np.random.randint(0, 1000, size=(INTEGRATION_SIZE, RANGES_SIZE), dtype=np.int64)
    gmf_tx = np.random.rand(INTEGRATION_SIZE)
    r_vec = np.random.rand(INTEGRATION_SIZE)
    v_vec = np.random.rand(INTEGRATION_SIZE)
    a_vec = np.random.rand(INTEGRATION_SIZE)
    g_vec = np.random.rand(INTEGRATION_SIZE)
    rgs = np.random.randint(0, 1000, size=(RANGES_SIZE), dtype=np.int32)
    fvec = np.random.rand(RANGE_RATE_SIZE)
    re = np.random.rand(MOCK_DIM_1, RANGE_RATE_SIZE)
    im = 1j * np.random.rand(MOCK_DIM_1, RANGE_RATE_SIZE)
    acceleration_phasors = re + im
    rx_stencil = np.random.choice([True, False], size=SAMPLE_SIZE)
    tx_stencil = np.random.choice([True, False], size=SAMPLE_SIZE)
    rx_window_indices = np.random.randint(0, 1000, size=(MOCK_DIM_2), dtype=np.int32)
    epoch_unix = np.float64(1.1)

    pointing_vec = np.tile([90.0, 75.0], (INTEGRATION_SIZE, 1))
    _pro = compute_process_params(
        exp_org,
        cfg_org,
        analysis_method=AnalysisMethod.target_estimation,
        method_lib=TargetEstimationMethod.fgmf,
        implementation=Impl.c,
    )
    # for target estimation the process parameters are done in two steps
    pro_org = GMFProcess.get_process_params(None, exp_org, cfg_org, _pro)
    pro_org = GMFProcess.get_lib_specific_process_params(None, exp_org, cfg_org, pro_org)

    gmf_out_args = MFOutArgs(
        num_cohints_per_file=cfg_org.num_cohints_per_file,
        ranges=RANGES_SIZE,
        range_rates=RANGE_RATE_SIZE,
        accelerations=ACCELERATIONS_SIZE,
        sample_numbers=SAMPLE_SIZE,
        vals=gmf_vals,
        dc=gmf_dc,
        v_ind=gmf_v_ind,
        a_ind=gmf_a_ind,
        tx_pwr=gmf_tx,
        snr=np.ndarray((10, 1), dtype=np.float64),
        r_vec=r_vec,
        v_vec=v_vec,
        a_vec=a_vec,
        g_vec=g_vec,
        pointing_vec=pointing_vec,
        epoch=epoch_unix,
        t=[1, 2, 3],
    )

    # Create a temporary file for gmf out
    with tempfile.TemporaryDirectory() as temp_dir:
        # mockup name for subfolder and gmf file
        outfile = Path(temp_dir) / "2015-10-22T11-00-00" / "mf-1445515198000000.h5"
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # dump
        h5_vars = GMFProcess.define_h5_vars(None, gmf_out_args).items()
        dump_params_to_file(h5_vars, exp_org, cfg_org, pro_org, outfile)

        # load
        out, exp, cfg, pro = list(load_analysed_data(temp_dir))[0]

        # validate out data
        out_d = out._asdict()
        for key, value in gmf_out_args._asdict().items():
            try:
                assert out_d[key] == value
            except ValueError:
                (out_d[key] == value).all()
        # validate exp data
        exp_org = exp_org._asdict()
        for key, value in exp._asdict().items():
            try:
                assert exp_org[key] == value
            except ValueError:
                (exp_org[key] == value).all()
        # validate config data
        for key, value in cfg_org.__dict__.items():
            try:
                assert cfg.__dict__[key] == value
            except ValueError:
                (cfg.__dict__[key] == value).all()
        # validate process data
        for key, value in pro_org.__dict__.items():
            try:
                assert pro.__dict__[key] == value
            except ValueError:
                (pro.__dict__[key] == value).all()
