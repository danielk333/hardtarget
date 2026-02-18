"""Discrete Polynomial-phase Transform, or DPT Process."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.fft as fft
from radardef.types import ExpParams
from scipy.signal import savgol_filter  # type: ignore[attr-defined]

from hardtarget.data_handling.configuration import extract_config_section
from hardtarget.matched_filter.dpt import get_dbt_lib
from hardtarget.matched_filter.dpt.types import DPTCfgParams, DPTProParams
from hardtarget.matched_filter.types import MFOutArgs, MFVariables
from hardtarget.process import Process
from hardtarget.types.constants import ConfigSubSection, Impl, TargetEstimationMethod
from hardtarget.types.types import AnalysisLib, CfgParams, DataItem, MethodLib, ProParams


class DPTProcess(Process[DPTCfgParams, DPTProParams, MFVariables, MFOutArgs, AnalysisLib]):
    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[AnalysisLib, TargetEstimationMethod, Impl]:
        return get_dbt_lib(lib, impl)

    def get_conf_params(self, cfg_path: Path, cfg_params: CfgParams) -> DPTCfgParams:
        """
        Extract DPT configuration parameters

        Args:
            cfg_path: Path to configuration file.
            cfg_params: already loaded configuraion parameters that can be extended

        Returns:
            Process specific DPT Configuration parameters
        """
        d = extract_config_section(
            cfg_path,
            ConfigSubSection.DPT,
            DPTCfgParams,
            cfg_params,
            self._logger,
        )
        return DPTCfgParams(**d)

    def get_process_params(
        self, exp_params: ExpParams, cfg_params: DPTCfgParams, pro_params: ProParams
    ) -> DPTProParams:
        """
        Calculate DPT specific process parameters

        Args:
            exp_params: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific DPT Process parameters
        """

        assert cfg_params.ipp_delay_parameter <= cfg_params.n_ipp, (
            f"ipp_delay_parameter: {cfg_params.ipp_delay_parameter} can not be larger than n_ipp: {cfg_params.n_ipp}"
        )

        # Sample times in the decimated il0d vector
        _rx_win_t = np.arange(pro_params.read_length) / exp_params.sample_rate
        _il0_rx_stencil_indices = np.argwhere(pro_params.rx_stencil).flatten()
        _rx_win_t = _rx_win_t[_il0_rx_stencil_indices]
        _rx_win_t = _rx_win_t[pro_params.il1_rx_window_indices]
        # take the decimated times as the center-points of decimated samples
        _rx_win_t_dec = np.mean(_rx_win_t.reshape(-1, cfg_params.frequency_decimation), axis=-1)
        # Time vector relative detected range-gate
        times2 = _rx_win_t_dec**2.0

        decimated_ipp_delay_parameter = np.floor(
            exp_params.ipp_samps * cfg_params.ipp_delay_parameter / cfg_params.frequency_decimation
        ).astype(np.int64)
        _step = 2 * decimated_ipp_delay_parameter * cfg_params.frequency_decimation / exp_params.sample_rate

        _max_accels_len = pro_params.decimated_read_length - decimated_ipp_delay_parameter
        accelerations = fft.fftshift(
            fft.fftfreq(_max_accels_len, d=cfg_params.frequency_decimation / exp_params.sample_rate)
        )
        accelerations = accelerations * exp_params.wavelength * 2 / _step  # m/s^2

        # filter accels
        _accel_inds = np.logical_and(
            accelerations >= cfg_params.min_acceleration, accelerations <= cfg_params.max_acceleration
        )
        inds_accelerations = np.argwhere(_accel_inds).flatten().astype(np.int32)

        # precalculate phasors corresponding to different accelerations
        acceleration_phasors = np.exp(
            -1j * np.pi * accelerations[:, None] * times2[None, :] / exp_params.wavelength
        ).astype(np.complex64)

        fgmf_acceleration_phasors = acceleration_phasors[_accel_inds, :].copy()

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

        tx, rx, ipp = self.get_data(start_sample, self.pro_params.read_length)

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
                self.cfg_params,
                self.pro_params,
            )
        else:
            return MFVariables(
                vals=np.zeros((len(self.pro_params.ranges),), dtype=np.float32),
                dc=np.zeros((len(self.pro_params.ranges),), dtype=np.float32),
                v_ind=np.zeros((len(self.pro_params.ranges),), dtype=np.int32),
                a_ind=np.zeros((len(self.pro_params.ranges),), dtype=np.int32),
                tx_pwr=np.array(tx_pwr),
            )

    def stack_vars(self, vars_list: list[MFVariables]) -> MFVariables:
        """Stack the results from the analysis"""
        return MFVariables(
            vals=np.stack([x.vals for x in vars_list], axis=0),
            dc=np.stack([x.dc for x in vars_list], axis=0),
            v_ind=np.stack([x.v_ind for x in vars_list], axis=0),
            a_ind=np.stack([x.a_ind for x in vars_list], axis=0),
            tx_pwr=np.stack([x.tx_pwr for x in vars_list], axis=0),
        )

    def generate_output(
        self,
        all_vars: MFVariables,
        file_idx_sample: int,
        exp_params: ExpParams,
        cfg_params: DPTCfgParams,
    ) -> MFOutArgs:
        """
        Calculate important parameters from the DPT analysis and generate the output.

        Args:
            all_vars: All cohints analysed data stacked together
            file_idx_sample: File id, microseconds since epoch.
            exp_params: Experiment parameters
            cfg_params: Configuration parameters

        Returns:
            Output data
        """
        sample_numbers = np.arange(self.pro_params.read_length, dtype=np.int32)
        num_cohints = all_vars.vals.shape[0]
        coh_ints = np.arange(num_cohints)

        # Substracting background level
        noise_floor = np.nanmedian(all_vars.dc, axis=0)
        noise_floor = savgol_filter(noise_floor, 2000, 1, mode="nearest")
        snr = (np.sqrt(all_vars.vals) - np.sqrt(noise_floor[None, :])) ** 2 / noise_floor[None, :]
        # finding peaks
        r_inds = np.argmax(snr, axis=1)

        r_vec = self.pro_params.ranges[r_inds]
        v_vec = self.pro_params.range_rates[all_vars.v_ind[coh_ints, r_inds]]
        a_vec = self.pro_params.accelerations[all_vars.a_ind[coh_ints, r_inds]]
        g_vec = all_vars.vals[coh_ints, r_inds]

        epoch_seconds = file_idx_sample * 1e-6

        _t_conv = (cfg_params.n_ipp * exp_params.t_ipp_usec) * 1e-6
        t = (np.arange(num_cohints) + 1) * _t_conv + epoch_seconds

        pointing_vec = np.zeros((num_cohints, 2), dtype=np.float32)
        for i in range(num_cohints):
            pointing = self.get_pointing(file_idx_sample + i * (cfg_params.n_ipp * exp_params.ipp_samps))
            pointing_vec[i, 0] = pointing.azimuth
            pointing_vec[i, 1] = pointing.elevation

        return MFOutArgs(
            num_cohints_per_file=num_cohints,
            ranges=self.pro_params.ranges,
            range_rates=self.pro_params.range_rates,
            accelerations=self.pro_params.accelerations,
            sample_numbers=sample_numbers,
            vals=all_vars.vals,
            dc=all_vars.dc,
            v_ind=all_vars.v_ind,
            a_ind=all_vars.a_ind,
            tx_pwr=all_vars.tx_pwr,
            snr=snr,
            r_vec=r_vec,
            v_vec=v_vec,
            a_vec=a_vec,
            g_vec=g_vec,
            pointing_vec=pointing_vec,
            t=t,
            epoch=epoch_seconds,
        )

    def define_h5_vars(self, output: MFOutArgs) -> dict[str, DataItem]:
        """
        Appends specifications to the MF output, such as dimensions, long names, units and more.

        Args:
            output: The output from the analysis

        Returns:
            A dictionary containing the output with attributes such as dimensions, long names and units.
        """

        str_dims_num_cohints_per_file = f"{output.num_cohints_per_file=}".split("=")[0].split(".")[1]
        str_t = f"{output.t=}".split("=")[0].split(".")[1]
        str_ranges = f"{output.ranges=}".split("=")[0].split(".")[1]
        return {
            str_dims_num_cohints_per_file: DataItem(
                data=output.num_cohints_per_file,
                long_name="number of cohints per file",
                scale=True,
            ),
            str_t: DataItem(
                data=output.t,
                long_name="time vector",
                scale=True,
            ),
            str_ranges: DataItem(
                data=output.ranges, long_name="Matched filter ranges", units="m", scale=True
            ),
            f"{output.range_rates=}".split("=")[0].split(".")[1]: DataItem(
                data=output.range_rates,
                long_name="Matched filter range rates",
                units="m/s",
                scale=True,
            ),
            f"{output.accelerations=}".split("=")[0].split(".")[1]: DataItem(
                data=output.accelerations,
                long_name="Matched filter range accelerations",
                units="m/s^2",
                scale=True,
            ),
            f"{output.sample_numbers=}".split("=")[0].split(".")[1]: DataItem(
                data=output.sample_numbers,
                long_name="Receiver sample number in radar cycle",
                scale=True,
            ),
            f"{output.vals=}".split("=")[0].split(".")[1]: DataItem(
                data=output.vals,
                dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                long_name="Generalized Matched Filter output values",
            ),
            f"{output.dc=}".split("=")[0].split(".")[1]: DataItem(
                data=output.dc,
                dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                long_name="Range dependant noise floor (0-frequency gmf output)",
            ),
            f"{output.v_ind=}".split("=")[0].split(".")[1]: DataItem(
                data=output.v_ind,
                dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                long_name="If range_rate is reduced, contains the best range rate index "
                "for each left over axis",
            ),
            f"{output.a_ind=}".split("=")[0].split(".")[1]: DataItem(
                data=output.a_ind,
                dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                long_name="If acceleration is reduced, contains the best acceleration "
                "index for each left over axis",
            ),
            f"{output.tx_pwr=}".split("=")[0].split(".")[1]: DataItem(
                data=output.tx_pwr,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="Transmitted signal power",
            ),
            f"{output.snr=}".split("=")[0].split(".")[1]: DataItem(
                data=output.snr,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="SNR for best range gate index",
            ),
            f"{output.r_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.r_vec,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="Range at peak GMF",
                units="m",
            ),
            f"{output.v_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.v_vec,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="Range rate at peak GMF",
            ),
            f"{output.a_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.a_vec,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="Acceleration at peak GMF",
            ),
            f"{output.g_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.g_vec, dims=[(str_dims_num_cohints_per_file, str_t)], long_name="Peak GMF"
            ),
            f"{output.pointing_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.pointing_vec,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="Radar pointing data (azimuth, elevation)",
            ),
            f"{output.epoch=}".split("=")[0].split(".")[1]: DataItem(
                data=output.epoch,
                long_name="epoch",  # TODO: better description
                scale=True,
            ),
        }
