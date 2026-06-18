"""General Matched Filter, or GMF Process"""

from dataclasses import asdict

import numpy as np
from radardef.types import ExpDef

from hardtarget.constants import ConfigSubSection, Impl, TargetEstimationMethod
from hardtarget.target_estimation.gmf import get_gmf_lib
from hardtarget.target_estimation.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.target_estimation.target_estimation_process import TargetEstimationProcess
from hardtarget.target_estimation.types import (
    MFVariables,
    TargetEstimationProParams,
)
from hardtarget.types import AnalysisLib, MethodLib


class GMFProcess(TargetEstimationProcess[GMFCfgParams, GMFProParams]):
    config_sub_section = ConfigSubSection.GMF

    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[AnalysisLib, TargetEstimationMethod, Impl]:
        return get_gmf_lib(lib, impl)

    def get_lib_specific_process_params(
        self,
        exp_def: ExpDef,
        cfg_params: GMFCfgParams,
        pro_params: TargetEstimationProParams,
    ) -> GMFProParams:
        """
        Calculate GMF specific process parameters

        Args:
            exp_def: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific GMF Process parameters
        """

        # Sample times in the decimated il0d vector
        _rx_win_t = np.arange(pro_params.read_length) / exp_def.sample_rate
        _il0_rx_stencil_indices = np.argwhere(pro_params.rx_stencil).flatten()
        _rx_win_t = _rx_win_t[_il0_rx_stencil_indices]
        _rx_win_t = _rx_win_t[pro_params.il1_rx_window_indices]
        # take the decimated times as the center-points of decimated samples
        _rx_win_t_dec = np.mean(_rx_win_t.reshape(-1, cfg_params.frequency_decimation), axis=-1)
        # Time vector relative detected range-gate
        times2 = _rx_win_t_dec**2.0

        accelerations = np.linspace(
            cfg_params.min_acceleration, cfg_params.max_acceleration, cfg_params.acceleration_steps
        )
        _accel_inds = np.logical_and(
            accelerations >= cfg_params.min_acceleration, accelerations <= cfg_params.max_acceleration
        )
        inds_accelerations = np.argwhere(_accel_inds).flatten().astype(np.int32)

        # precalculate phasors corresponding to different accelerations
        acceleration_phasors = np.exp(
            -1j * np.pi * np.outer(accelerations, times2) / exp_def.wavelength
        ).astype(np.complex64)

        fgmf_acceleration_phasors = acceleration_phasors[_accel_inds, :]

        return GMFProParams(
            **asdict(pro_params),
            accelerations=accelerations,
            inds_accelerations=inds_accelerations,
            acceleration_phasors=acceleration_phasors,
            fgmf_acceleration_phasors=fgmf_acceleration_phasors,
        )

    def analyse_ipps(self, start_sample: int) -> MFVariables:
        """
        Analyse the interpulse periods from start sample with the choosen GMF method.

        Args:
            start_sample: sample index to start analysis at.


        Returns:
            Outcome of GMF analysis
        """

        tx, rx, ipp = self.get_data(
            start_sample,
            self.pro_params.read_length,
            sub_resolution=self.cfg_params.range_gate_sub_resolution,
        )

        # TODO: generalize a preprocess filtering of 0 tx power
        # since it can cause unnessary slowdowns depending on experiment setup
        # e.g. a tx signal with multiple pulses with pauses between can be faster
        # computed by skipping the 0-tx periods inside the tx interval

        # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
        # scale transmit waveform to unity power
        tx_pwr = np.sum(np.abs(tx) ** 2.0)
        tx_amp = np.sqrt(tx_pwr)
        tx = np.conj(tx) / tx_amp

        if tx_amp > self.cfg_params.tx_amp_limit:
            kwargs = {}
            if self.pro_params.implementation == Impl.cuda:
                kwargs["gpu_id"] = 1 % self.cfg_params.node_gpus  # TODO:1 should be job.idx

            return self.lib(tx, rx, np.array(tx_pwr), self.cfg_params, self.pro_params, **kwargs)
        else:
            return MFVariables(
                vals=np.zeros((len(self.pro_params.ranges),), dtype=np.float32),
                dc=np.zeros((len(self.pro_params.ranges),), dtype=np.float32),
                v_ind=np.zeros((len(self.pro_params.ranges),), dtype=np.int32),
                a_ind=np.zeros((len(self.pro_params.ranges),), dtype=np.int32),
                tx_pwr=np.array(tx_pwr),
            )
