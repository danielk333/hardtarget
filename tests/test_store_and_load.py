import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import radardef

from hardtarget.constants import AnalysisMethod
from hardtarget.data_handling import compute_process_params, dump_params_to_file
from hardtarget.data_handling.configuration import extract_config_params_from_derived_object
from hardtarget.event_detection.types import XCorrCfgParams, XCorrOutArgs, XCorrProParams
from hardtarget.interferometry.types import DOACfgParams, DOAOutArgs, DOAProParams
from hardtarget.optimization.types import MFOptimizeOutArgs, OptimizeCfgParams, OptimizeProParams
from hardtarget.plotting.load_data import get_process_types, load_analysed_data, orig_bases
from hardtarget.process import (
    DOAProcess,
    DPTProcess,
    GMFProcess,
    OptimizeProcess,
    Process,
    TargetEstimationProcess,
    XCorrProcess,
    get_analysis_process,
)
from hardtarget.target_estimation.dpt.types import DPTCfgParams, DPTProParams
from hardtarget.target_estimation.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.target_estimation.types import MFOutArgs
from hardtarget.types import ArrayKwargs, CfgParams, ExpParams


class TestStoreAndLoad:
    method_and_process = [
        (AnalysisMethod.target_estimation, GMFProcess),
        (AnalysisMethod.optimize, OptimizeProcess),
        (AnalysisMethod.event_detection, XCorrProcess),
        (AnalysisMethod.direction_of_arrival, DOAProcess),
    ]

    @pytest.mark.parametrize("method, expected_process", method_and_process)
    def test_get_process(self, method, expected_process):
        process = get_analysis_process(method)
        assert expected_process is process, f"{process.__name__} is not comptible with {method}"

    process_and_types = [
        (GMFProcess, GMFCfgParams, GMFProParams, MFOutArgs),
        (DPTProcess, DPTCfgParams, DPTProParams, MFOutArgs),
        (OptimizeProcess, OptimizeCfgParams, OptimizeProParams, MFOptimizeOutArgs),
        (XCorrProcess, XCorrCfgParams, XCorrProParams, XCorrOutArgs),
        (DOAProcess, DOACfgParams, DOAProParams, DOAOutArgs),
    ]

    @pytest.mark.parametrize("process, cfg_type, pro_type, out_type", process_and_types)
    def test_get_process_types(self, process, cfg_type, pro_type, out_type):

        cfg, pro, out = get_process_types(process)

        assert cfg is cfg_type, (
            f"{process.__name__}: {cfg.__name__} is not the expected cfg type: {cfg_type.__name__}"
        )
        assert pro is pro_type, (
            f"{process.__name__}: {pro.__name__} is not the expected pro type: {pro_type.__name__}"
        )
        assert out is out_type, (
            f"{process.__name__}: {out.__name__} is not the expected out type: {out_type.__name__}"
        )

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

    @pytest.mark.parametrize("process", [GMFProcess, DPTProcess])
    def test_target_estimation_data(self, process: type[TargetEstimationProcess]):

        cfg_type, _ = orig_bases(process)[0].__args__
        cfg_org = cfg_type(n_ipp=2)
        t_size = 10
        out_org = MFOutArgs(
            num_cohints_per_file=cfg_org.num_cohints_per_file,
            ranges=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            range_rates=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            accelerations=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            sample_numbers=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int32),
            vals=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.float32),
            dc=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.float32),
            v_ind=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.int32),
            a_ind=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.int32),
            tx_pwr=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            snr=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.float64),
            r_vec=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.float64),
            v_vec=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.float64),
            a_vec=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.float64),
            g_vec=np.random.rand(cfg_org.num_cohints_per_file, t_size).astype(np.float32),
            pointing_vec=np.random.rand(cfg_org.num_cohints_per_file, 2).astype(np.float32),
            epoch=100.0,
            t=np.random.rand(t_size).astype(np.float32),
        )

        self.store_and_load(self.exp_org, cfg_org, out_org, process, AnalysisMethod.target_estimation)

    def test_event_detection(self):

        cfg_org = XCorrCfgParams()
        out_org = XCorrOutArgs(
            max_pow=np.random.rand(cfg_org.num_cohints_per_file).astype(np.complex128),
            max_pow_norm=np.random.rand(cfg_org.num_cohints_per_file).astype(np.complex128),
            max_peak=np.random.rand(cfg_org.num_cohints_per_file).astype(np.complex128),
            max_pow_ind=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            best_doppler=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            ipps_pow=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
        )

        self.store_and_load(self.exp_org, cfg_org, out_org, XCorrProcess, AnalysisMethod.event_detection)

    def test_direction_of_arrival(self):
        cfg_org = DOACfgParams()
        out_org = DOAOutArgs(
            vals=np.random.rand(cfg_org.num_cohints_per_file, 25, 25).astype(np.complex64),
            k_vec=np.random.rand(cfg_org.num_cohints_per_file, 3).astype(np.float32),
            peak=np.random.rand(cfg_org.num_cohints_per_file).astype(np.complex64),
            azimuth=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float32),
            elevation=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float32),
        )

        array_kwargs = ArrayKwargs(beam=radardef.Mu().beam, parameters=radardef.Mu().beam_parameters)

        self.store_and_load(
            self.exp_org, cfg_org, out_org, DOAProcess, AnalysisMethod.direction_of_arrival, **array_kwargs
        )

    def store_and_load(
        self,
        exp_org: ExpParams,
        cfg_org: CfgParams,
        out_org: NamedTuple,
        process: type[Process],
        method: AnalysisMethod,
        **kwargs: ArrayKwargs,
    ):
        _cfg = extract_config_params_from_derived_object(cfg_org)
        _pro = compute_process_params(
            exp_org,
            _cfg,
            analysis_method=method,
        )
        _process = process(cfg_org, exp_org, _cfg, _pro, None, None, None, **kwargs)
        pro_org = _process.pro_params

        # Create a temporary file for gmf out
        with tempfile.TemporaryDirectory() as temp_dir:
            # mockup name for subfolder and gmf file
            outfile = Path(temp_dir) / "2015-10-22T11-00-00" / "mf-1445515198000000.h5"
            outfile.parent.mkdir(parents=True, exist_ok=True)

            # store data
            h5_vars = _process.define_h5_vars(out_org).items()
            dump_params_to_file(h5_vars, exp_org, cfg_org, pro_org, outfile)

            # load
            out, exp, cfg, pro = list(load_analysed_data(temp_dir))[0]

            # Validate data types
            assert type(out) is type(out_org), (
                f"Out is not of the correct type, org: {type(out_org)}, loaded: {type(out)}"
            )
            assert type(exp) is type(exp_org), (
                f"Exp is not of the correct type, org: {type(exp_org)}, loaded: {type(exp)}"
            )
            assert type(cfg) is type(cfg_org), (
                f"Cfg is not of the correct type, org: {type(pro_org)}, loaded: {type(cfg)}"
            )
            assert type(pro) is type(pro_org), (
                f"Out is not of the correct type, org: {type(pro_org)}, loaded: {type(pro)}"
            )

            # validate out data
            out_d = out._asdict()
            for key, value in out_org._asdict().items():
                try:
                    assert out_d[key] == value
                except ValueError:
                    (out_d[key] == value).all()
            # validate exp data
            exp_d = exp._asdict()
            for key, value in exp_org._asdict().items():
                try:
                    assert exp_d[key] == value
                except ValueError:
                    (exp_d[key] == value).all()
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
