"""Generic Target Estimation process"""

import sys
from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import scipy.fft as fft
from scipy.signal import savgol_filter  # type: ignore[attr-defined]

from hardtarget.constants import AnalysisMethod, ConfigSubSection
from hardtarget.process import Process
from hardtarget.process.configuration import extract_config_section, get_ilx_windows
from hardtarget.target_estimation.types import (
    ExtendedTargetEstimationProParams,
    MFOutArgs,
    MFVariables,
    TargetEstimationCfgParams,
    TargetEstimationProParams,
)
from hardtarget.types import (
    DataItem,
    ExpDef,
    ProParams,
    TargetEstimationLib,
)
from hardtarget.utils import noise
from hardtarget.utils.range_conversion import range_gate_to_range

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    pass
else:
    pass

TeLibCfg = TypeVar("TeLibCfg", bound=TargetEstimationCfgParams)
TeLibPro = TypeVar("TeLibPro", bound=ExtendedTargetEstimationProParams)


class TargetEstimationProcess(
    Process[
        TargetEstimationCfgParams,
        TargetEstimationProParams,
        MFVariables,
        MFOutArgs,
        TargetEstimationLib[TeLibCfg, TeLibPro, MFVariables],
    ],
    Generic[TeLibCfg, TeLibPro],
):
    method = AnalysisMethod.target_estimation
    config_section = ConfigSubSection.TARGET_ESTIMATION
    config_sub_section: ConfigSubSection

    def __post_init__(self) -> None:
        """Extract configuration parameters for the specfic child class"""

        if isinstance(self.raw_config, TargetEstimationCfgParams):
            self.cfg_params: TeLibCfg = self.raw_config  # type: ignore[assignment]
        elif not isinstance(self.raw_config, dict):
            self.cfg_params = self.get_lib_specific_conf_params(Path(self.raw_config), self.cfg_params)
        self.pro_params: TeLibPro = self.get_lib_specific_process_params(
            self.exp_def, self.cfg_params, self.pro_params
        )

    def get_lib_specific_conf_params(self, cfg_path: Path, cfg_params: TargetEstimationCfgParams) -> TeLibCfg:

        cfg_type, _, _ = self.get_types()
        d = extract_config_section(
            cfg_path,
            self.config_sub_section,
            cfg_type,
            cfg_params,
            self._logger,
        )

        return cfg_type(**d)  # type: ignore[return-value]

    def get_process_params(
        self, exp_def: ExpDef, cfg_params: TargetEstimationCfgParams, pro_params: ProParams
    ) -> TargetEstimationProParams:
        """
        Calculate Optimize specific process parameters

        Args:
            exp_def: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific Optimize Process parameters
        """

        def usec_to_samp(usec: int | float) -> int:
            return int(usec / exp_def.t_samp_usec)

        rx_start_samp = usec_to_samp(exp_def.t_rx_start_usec)
        rx_end_samp = usec_to_samp(exp_def.t_rx_end_usec)
        tx_start_samp = usec_to_samp(exp_def.t_tx_start_usec)
        tx_end_samp = usec_to_samp(exp_def.t_tx_end_usec)

        decimated_read_length = np.ceil(pro_params.read_length / cfg_params.frequency_decimation).astype(
            np.int64
        )

        full_res_rgs = np.arange(
            pro_params.range_gates[0],
            pro_params.range_gates[-1] + cfg_params.range_gate_step,
            cfg_params.range_gate_step / cfg_params.range_gate_sub_resolution,
            dtype=np.float64,
        )

        ranges = range_gate_to_range(full_res_rgs + 1, exp_def.sample_rate).astype(np.float64)  # m

        assert np.all(pro_params.range_gates >= 0), "Computed range gates not compatible with stencils"

        _tx_pulse_samps = tx_end_samp - tx_start_samp

        il0_rgs, il0_rx_window_indices, il1_rx_window_indices = get_ilx_windows(
            exp_def, cfg_params, pro_params
        )

        assert il0_rgs.max() <= rx_end_samp - _tx_pulse_samps, (
            f"end range gate {il0_rgs.max()} cannot be after RX end (-minus tx length) {rx_end_samp - _tx_pulse_samps}"
        )
        assert il0_rgs.max() > rx_start_samp, (
            f"end range gate: {il0_rgs.max()} cannot be before RX start: {rx_start_samp}"
        )
        assert il0_rgs.min() <= rx_end_samp - _tx_pulse_samps, "start range gate cannot be after  RX end"
        assert il0_rgs.min() >= rx_start_samp, (
            f"start range gate: {il0_rgs.min()}  cannot be before RX start: {rx_start_samp}"
        )
        _il0_tx_stencil_indices = np.argwhere(pro_params.tx_stencil).flatten()

        _coh_int_samps = len(_il0_tx_stencil_indices)

        # Decimated signals
        # TODO: this can probably be allowed if we truncate/pad the end or start of the decimated vector
        assert _tx_pulse_samps % cfg_params.frequency_decimation == 0, (
            "Pulse samples should be divisible by decimation to avoid edge effects\n"
            f"tx_pulse_samps / frequency_decimation = {_tx_pulse_samps}/{cfg_params.frequency_decimation} ="
            f"{_tx_pulse_samps / cfg_params.frequency_decimation}"
        )
        # TODO: this can be avoided by padding the stencil and having a second 0-stencil
        # TODO: better assert messages if keep
        assert len(ranges) % cfg_params.frequency_decimation == 0, (
            "range-gate interval not compatible with decimation: "
            f"{len(ranges)} % {cfg_params.frequency_decimation} = "
            f"{len(ranges) % cfg_params.frequency_decimation}"
        )
        assert exp_def.ipp_samps % cfg_params.frequency_decimation == 0, (
            "ipp-samples length not compatible with decimation: "
            f"{exp_def.ipp_samps} % {cfg_params.frequency_decimation} = "
            f"{exp_def.ipp_samps % cfg_params.frequency_decimation}"
        )
        assert _coh_int_samps % cfg_params.frequency_decimation == 0, (
            "range-gate interval not compatible with decimation: "
            f"{_coh_int_samps} % {cfg_params.frequency_decimation} = "
            f"{_coh_int_samps % cfg_params.frequency_decimation}"
        )

        il0_dec_rx_window_indices = (
            il0_rx_window_indices[:: cfg_params.frequency_decimation] // cfg_params.frequency_decimation
        ).astype(np.int32)

        # ---- Velocity related parameters ----

        # frequency vector
        fft_frequencies = fft.fftshift(
            fft.fftfreq(
                decimated_read_length,
                d=cfg_params.frequency_decimation / exp_def.sample_rate,
            )
        )  # Hz

        range_rates = (exp_def.wavelength * fft_frequencies).astype(np.float64)

        return TargetEstimationProParams(
            **asdict(pro_params),
            decimated_read_length=decimated_read_length,
            il0_rgs=il0_rgs,
            ranges=ranges,
            il1_rx_window_indices=il1_rx_window_indices,
            il0_rx_window_indices=il0_rx_window_indices,
            il0_dec_rx_window_indices=il0_dec_rx_window_indices,
            range_rates=range_rates,
            fft_frequencies=fft_frequencies,
        )

    @abstractmethod
    def get_lib_specific_process_params(
        self,
        exp_def: ExpDef,
        cfg_params: TeLibCfg,
        pro_params: TargetEstimationProParams,
    ) -> TeLibPro:
        pass

    def stack_vars(self, vars_list: list[MFVariables]) -> MFVariables:
        """Stack the results from the analysis"""

        return MFVariables(
            vals=np.stack([x.vals for x in vars_list], axis=0),
            dc=np.stack([x.dc for x in vars_list], axis=0),
            v=np.stack([x.v for x in vars_list], axis=0),
            a=np.stack([x.a for x in vars_list], axis=0),
            phi=np.stack([x.phi for x in vars_list], axis=0),
            tx_pwr=np.stack([x.tx_pwr for x in vars_list], axis=0),
        )

    def generate_output(
        self,
        all_vars: MFVariables,
        file_idx_sample: int,
        exp_def: ExpDef,
        cfg_params: TargetEstimationCfgParams,
        pro_params: TargetEstimationProParams,
    ) -> MFOutArgs:
        """
        Calculate important parameters from the GMF analysis and generate the output.

        Args:
            all_vars: All cohints analysed data stacked together
            file_idx_sample: File id, sample relative to file start.
            exp_def: Experiment parameters
            cfg_params: Configuration parameters

        Returns:
            Output data
        """
        sample_numbers = np.arange(pro_params.read_length, dtype=np.int32)
        num_cohints = all_vars.vals.shape[0]
        coh_ints = np.arange(num_cohints)

        # Substracting background level
        noise_floor = np.nanmedian(all_vars.dc, axis=0)
        noise_floor = savgol_filter(noise_floor, 2000, 1, mode="nearest")

        # Calculating signal to noise ratio
        snr = noise.snr(all_vars.vals, noise_floor)

        # finding peaks
        r_inds = np.argmax(snr, axis=1)
        snr_vec = snr[coh_ints, r_inds]
        r_vec = pro_params.ranges[r_inds]
        v_vec = all_vars.v[coh_ints, r_inds]
        a_vec = all_vars.a[coh_ints, r_inds]
        g_vec = all_vars.vals[coh_ints, r_inds]

        epoch_us = int(self.data.epoch_bounds[0] + file_idx_sample * exp_def.t_samp_usec)

        _t_conv = (cfg_params.n_ipp * exp_def.t_ipp_usec) * 1e-6
        t = (np.arange(num_cohints) + 1) * _t_conv + file_idx_sample * exp_def.t_samp_usec * 1e-6

        pointing_vec = np.zeros((num_cohints, 2), dtype=np.float64)
        for i in range(num_cohints):
            pointing = self.get_pointing(file_idx_sample + i * (cfg_params.n_ipp * exp_def.ipp_samps))
            pointing_vec[i, 0] = pointing.azimuth
            pointing_vec[i, 1] = pointing.elevation

        return MFOutArgs(
            num_cohints_per_file=num_cohints,
            ranges=pro_params.ranges,
            range_rates=pro_params.range_rates,
            accelerations=pro_params.accelerations,  # type: ignore[attr-defined]
            sample_numbers=sample_numbers,
            vals=all_vars.vals,
            dc=all_vars.dc,
            tx_pwr=all_vars.tx_pwr,
            snr=snr,
            v=all_vars.v,
            a=all_vars.a,
            phi=all_vars.phi,
            snr_vec=snr_vec,
            r_vec=r_vec,
            v_vec=v_vec,
            a_vec=a_vec,
            g_vec=g_vec,
            pointing_vec=pointing_vec,
            t=t,
            epoch_us=epoch_us,
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
        # TODO: add phase to output
        return {
            str_dims_num_cohints_per_file: DataItem(
                data=output.num_cohints_per_file,
                long_name="Number of cohints per file",
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
            f"{output.v=}".split("=")[0].split(".")[1]: DataItem(
                data=output.v,
                dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                # TODO: update the long names descriptions, we no longer allow output
                # with no reduction
                long_name="If range_rate is reduced, contains the best range rate for each left over axis",
            ),
            f"{output.a=}".split("=")[0].split(".")[1]: DataItem(
                data=output.a,
                dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                long_name="If acceleration is reduced, contains the best acceleration "
                "index for each left over axis",
            ),
            f"{output.phi=}".split("=")[0].split(".")[1]: DataItem(
                data=output.a,
                dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                long_name="TODO",
            ),
            f"{output.tx_pwr=}".split("=")[0].split(".")[1]: DataItem(
                data=output.tx_pwr,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="Transmitted signal power",
            ),
            f"{output.snr=}".split("=")[0].split(".")[1]: DataItem(
                data=output.snr,
                dims=[(str_dims_num_cohints_per_file, str_t)],
                long_name="SNR for all gates",
            ),
            f"{output.snr_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.snr_vec,
                dims=[(str_dims_num_cohints_per_file, "1")],
                long_name="SNR at peak GMF",
            ),
            f"{output.r_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.r_vec,
                dims=[(str_dims_num_cohints_per_file, "1")],
                long_name="Range at peak GMF",
            ),
            f"{output.v_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.v_vec,
                dims=[(str_dims_num_cohints_per_file, "1")],
                long_name="Range rate at peak GMF",
            ),
            f"{output.a_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.a_vec,
                dims=[(str_dims_num_cohints_per_file, "1")],
                long_name="Acceleration at peak GMF",
            ),
            f"{output.g_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.g_vec, dims=[(str_dims_num_cohints_per_file, str_t)], long_name="Peak GMF"
            ),
            f"{output.pointing_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.pointing_vec,
                dims=[(str_dims_num_cohints_per_file, "2")],
                long_name="Radar pointing data (azimuth, elevation)",
            ),
            f"{output.epoch_us=}".split("=")[0].split(".")[1]: DataItem(
                data=output.epoch_us,
                long_name="Epoch of the first analysed datapoint in microseconds",
                scale=True,
            ),
        }
