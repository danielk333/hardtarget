"""Discrete Polynomial-phase Transform, or DPT Process."""

from dataclasses import asdict

import numpy as np
import scipy.fft as fft
from radardef.types import ExpDef

from hardtarget.constants import ConfigSubSection, Impl, TargetEstimationMethod
from hardtarget.target_estimation.dpt import get_dbt_lib
from hardtarget.target_estimation.dpt.types import DPTCfgParams, DPTProParams
from hardtarget.target_estimation.target_estimation_process import TargetEstimationProcess
from hardtarget.target_estimation.types import (
    MFVariables,
    TargetEstimationProParams,
)
from hardtarget.types import MethodLib, TargetEstimationLib


class DPTProcess(TargetEstimationProcess[DPTCfgParams, DPTProParams]):
    config_sub_section = ConfigSubSection.DPT

    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[TargetEstimationLib, TargetEstimationMethod, Impl]:
        return get_dbt_lib(lib, impl)

    def get_lib_specific_process_params(
        self,
        exp_def: ExpDef,
        cfg_params: DPTCfgParams,
        pro_params: TargetEstimationProParams,
    ) -> DPTProParams:
        """
        Calculate DPT specific process parameters

        Args:
            exp_def: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific DPT Process parameters
        """

        assert cfg_params.ipp_delay_parameter < cfg_params.n_ipp, (
            f"ipp_delay_parameter: {cfg_params.ipp_delay_parameter} can not be larger or equal to n_ipp: {cfg_params.n_ipp}"
        )

        # Sample times in the decimated il0d vector
        _rx_win_t = np.arange(pro_params.read_length) / exp_def.sample_rate
        _il0_rx_stencil_indices = np.argwhere(pro_params.rx_stencil).flatten()
        _rx_win_t = _rx_win_t[_il0_rx_stencil_indices]
        _rx_win_t = _rx_win_t[pro_params.il1_rx_window_indices]
        # take the decimated times as the center-points of decimated samples
        _rx_win_t_dec = np.mean(_rx_win_t.reshape(-1, cfg_params.frequency_decimation), axis=-1)
        # Time vector relative detected range-gate
        times2 = _rx_win_t_dec**2.0

        decimated_ipp_delay_parameter = np.floor(
            exp_def.ipp_samps * cfg_params.ipp_delay_parameter / cfg_params.frequency_decimation
        ).astype(np.int64)
        _step = 2 * decimated_ipp_delay_parameter * cfg_params.frequency_decimation / exp_def.sample_rate

        _max_accels_len = pro_params.decimated_read_length - decimated_ipp_delay_parameter
        accelerations = fft.fftshift(
            fft.fftfreq(_max_accels_len, d=cfg_params.frequency_decimation / exp_def.sample_rate)
        )
        accelerations = accelerations * exp_def.wavelength * 2 / _step  # m/s^2

        # filter accels
        _accel_inds = np.logical_and(
            accelerations >= cfg_params.min_acceleration, accelerations <= cfg_params.max_acceleration
        )
        inds_accelerations = np.argwhere(_accel_inds).flatten().astype(np.int32)

        # precalculate phasors corresponding to different accelerations
        # TODO: this size problem can be fixed by only computing the ones inside _accel_inds
        # this might even be removed since its easy to compute just one for the amplitude and phase
        # computation, we are not looping trough them like with gmf
        acceleration_phasors = np.exp(
            -1j * np.pi * np.outer(accelerations, times2) / exp_def.wavelength
        ).astype(np.complex64)

        fgmf_acceleration_phasors = acceleration_phasors[_accel_inds, :]

        return DPTProParams(
            **asdict(pro_params),
            decimated_ipp_delay_parameter=decimated_ipp_delay_parameter,
            inds_accelerations=inds_accelerations,
            accelerations=accelerations,
            acceleration_phasors=acceleration_phasors,
            fgmf_acceleration_phasors=fgmf_acceleration_phasors,
        )

    def analyse_ipps(self, start_sample: int) -> MFVariables:
        """
        Analyse the interpulse periods from start sample with the choosen DPT method.

        Args:
            start_sample: sample index to start analysis at.

        Returns:
            Outcome of DPT analysis
        """

        tx, rx, ipp = self.get_data(
            start_sample,
            self.pro_params.read_length,
            sub_resolution=self.cfg_params.range_gate_sub_resolution,
        )

        if len(rx) == 0:
            raise ValueError("No data on rx signal")
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
            return self.lib(
                tx,
                rx,
                np.array(tx_pwr),
                self.exp_def,
                self.cfg_params,
                self.pro_params,
            )
        else:
            return MFVariables(
                vals=np.zeros((len(self.pro_params.ranges),), dtype=np.float32),
                dc=np.zeros((len(self.pro_params.ranges),), dtype=np.float32),
                v=np.zeros((len(self.pro_params.ranges),), dtype=np.float64),
                a=np.zeros((len(self.pro_params.ranges),), dtype=np.float64),
                phi=np.zeros((len(self.pro_params.ranges),), dtype=np.float64),
                tx_pwr=np.array(tx_pwr),
            )
