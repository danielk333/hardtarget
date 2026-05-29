import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import radardef
from radardef.radar_stations.eiscat.experiments import leo_mpark_2_1u

from hardtarget.constants import AnalysisMethod, DOAMethod, EchoSearchMethod, Impl, TargetEstimationMethod
from hardtarget.data_handling import dump_params_to_file
from hardtarget.echo_search.types import EchoSearchCfgParams, EchoSearchOutArgs, EchoSearchProParams
from hardtarget.interferometry.types import DOACfgParams, DOAOutArgs, DOAProParams
from hardtarget.optimization.types import MFOptimizeOutArgs, OptimizeCfgParams, OptimizeProParams
from hardtarget.plotting.load_data import load_analysed_data, stack_analysed_data
from hardtarget.process import (
    DOAProcess,
    DPTProcess,
    EchoSearchProcess,
    GMFProcess,
    OptimizeProcess,
    Process,
    TargetEstimationProcess,
    compute_process_params,
    get_analysis_process,
)
from hardtarget.process.configuration import extract_config_params_from_derived_object
from hardtarget.target_estimation.dpt.types import DPTCfgParams, DPTProParams
from hardtarget.target_estimation.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.target_estimation.types import MFOutArgs
from hardtarget.types import AnalysedResult, ArrayKwargs, CfgParams, ExpDef, MethodLib
from src.hardtarget.constants import MethodAbbreviation


class TestStoreAndLoad:
    method_and_process = [
        (AnalysisMethod.target_estimation, GMFProcess),
        (AnalysisMethod.optimize, OptimizeProcess),
        (AnalysisMethod.echo_search, EchoSearchProcess),
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
        (EchoSearchProcess, EchoSearchCfgParams, EchoSearchProParams, EchoSearchOutArgs),
        (DOAProcess, DOACfgParams, DOAProParams, DOAOutArgs),
    ]

    @pytest.mark.parametrize("process, cfg_type, pro_type, out_type", process_and_types)
    def test_get_process_types(self, process, cfg_type, pro_type, out_type):

        cfg, pro, out = process.get_types()

        assert cfg is cfg_type, (
            f"{process.__name__}: {cfg.__name__} is not the expected cfg type: {cfg_type.__name__}"
        )
        assert pro is pro_type, (
            f"{process.__name__}: {pro.__name__} is not the expected pro type: {pro_type.__name__}"
        )
        assert out is out_type, (
            f"{process.__name__}: {out.__name__} is not the expected out type: {out_type.__name__}"
        )

    exp_org = leo_mpark_2_1u

    @pytest.mark.parametrize(
        "process, lib", [(GMFProcess, TargetEstimationMethod.fgmf), (DPTProcess, TargetEstimationMethod.fdpt)]
    )
    def test_target_estimation_data(
        self, process: type[TargetEstimationProcess], lib: TargetEstimationMethod
    ):

        cfg_type, _, _ = process.get_types()
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
            epoch_us=100,
            t=np.random.rand(t_size).astype(np.float32),
        )

        self.store_and_load(
            self.exp_org,
            cfg_org,
            out_org,
            process,
            AnalysisMethod.target_estimation,
            lib,
            Impl.c,
        )

    def test_echo_search(self):

        cfg_org = EchoSearchCfgParams()
        out_org = EchoSearchOutArgs(
            max_corr=np.random.rand(cfg_org.num_cohints_per_file).astype(np.complex128),
            max_corr_ind=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            max_corr_delay=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            best_doppler=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            tot_pow=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            mean=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            std_dev=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            epoch_us=int(np.random.rand()),
        )

        self.store_and_load(
            self.exp_org,
            cfg_org,
            out_org,
            EchoSearchProcess,
            AnalysisMethod.echo_search,
            EchoSearchMethod.xcorr,
            Impl.numpy,
        )

    def test_direction_of_arrival(self):
        cfg_org = DOACfgParams()
        out_org = DOAOutArgs(
            k_vec=np.random.rand(cfg_org.num_cohints_per_file, 3).astype(np.float32),
            peak=np.random.rand(cfg_org.num_cohints_per_file).astype(np.complex64),
            azimuth=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float32),
            elevation=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float32),
            epoch_us=int(np.random.rand()),
        )

        array_kwargs = ArrayKwargs(beam=radardef.Mu().beam, parameters=radardef.Mu().beam_parameters)

        self.store_and_load(
            self.exp_org,
            cfg_org,
            out_org,
            DOAProcess,
            AnalysisMethod.direction_of_arrival,
            DOAMethod.music_grid_search,
            Impl.numpy,
            **array_kwargs,
        )

    def store_and_load(
        self,
        exp_org: ExpDef,
        cfg_org: CfgParams,
        out_org: NamedTuple,
        process: type[Process],
        method: AnalysisMethod,
        method_lib: MethodLib,
        impl: Impl,
        **kwargs: ArrayKwargs,
    ):
        _cfg = extract_config_params_from_derived_object(cfg_org)
        _pro = compute_process_params(
            exp_org, _cfg, analysis_method=method, method_lib=method_lib, implementation=impl
        )

        pro_org = process.get_process_params(None, exp_org, cfg_org, _pro)
        if issubclass(process, TargetEstimationProcess):
            pro_org = process.get_lib_specific_process_params(None, exp_org, cfg_org, pro_org)

        # Create a temporary file for gmf out
        with tempfile.TemporaryDirectory() as temp_dir:
            # mockup name for subfolder and gmf file
            outfile = (
                Path(temp_dir) / "2015-10-22T11-00-00" / f"{MethodAbbreviation[method]}-1445515198000000.h5"
            )
            outfile.parent.mkdir(parents=True, exist_ok=True)

            # store data
            h5_vars = process.define_h5_vars(None, out_org).items()
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
            for key, value in exp_org.__dict__.items():
                try:
                    assert exp.__dict__[key] == value
                except ValueError:
                    (exp.__dict__[key] == value).all()
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

    def test_stack_data_from_ram(self):
        cfg_org = EchoSearchCfgParams()
        out_org = EchoSearchOutArgs(
            max_corr=np.random.rand(cfg_org.num_cohints_per_file).astype(np.complex128),
            max_corr_ind=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            max_corr_delay=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            best_doppler=np.random.rand(cfg_org.num_cohints_per_file).astype(np.int64),
            tot_pow=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            mean=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            std_dev=np.random.rand(cfg_org.num_cohints_per_file).astype(np.float64),
            epoch_us=int(np.random.rand()),
        )

        _cfg = extract_config_params_from_derived_object(cfg_org)
        _pro = compute_process_params(
            self.exp_org,
            _cfg,
            analysis_method=AnalysisMethod.echo_search,
            method_lib=EchoSearchMethod.xcorr,
            implementation=Impl.c,
        )

        pro_org = EchoSearchProcess.get_process_params(None, self.exp_org, cfg_org, _pro)

        analysed_results: AnalysedResult = {
            "dir": None,
            "files": [],
            "data": {
                1: (out_org, self.exp_org, cfg_org, pro_org),
                2: (out_org, self.exp_org, cfg_org, pro_org),
                3: (out_org, self.exp_org, cfg_org, pro_org),
            },
        }

        out, exp, cfg, pro = stack_analysed_data(analysed_results["data"])

        assert len(out.max_corr) == 3 * cfg_org.num_cohints_per_file
        assert len(out.max_corr_ind) == 3 * cfg_org.num_cohints_per_file
        assert len(out.best_doppler) == 3 * cfg_org.num_cohints_per_file
        assert len(out.tot_pow) == 3 * cfg_org.num_cohints_per_file

        assert exp == self.exp_org
        assert cfg_org == cfg
        assert pro_org == pro
