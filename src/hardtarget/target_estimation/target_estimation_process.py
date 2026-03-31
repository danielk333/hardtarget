"""Generic Target Estimation process"""

import sys
from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Generic, TypeVar

import numpy as np
import scipy.fft as fft
from radardef.types import Pointing
from scipy.signal import savgol_filter  # type: ignore[attr-defined]

from hardtarget.constants import ConfigSubSection
from hardtarget.data_handling.configuration import (
    extract_config_section,
    get_ilx_windows,
)
from hardtarget.process import Process
from hardtarget.target_estimation.types import (
    ExtendedTargetEstimationProParams,
    MFOutArgs,
    MFVariables,
    TargetEstimationCfgParams,
    TargetEstimationProParams,
)
from hardtarget.types import (
    AnalysisLib,
    ArrayKwargs,
    Bounds,
    CfgParams,
    DataItem,
    ExpDef,
    ExtractSignals,
    ProParams,
)
from hardtarget.utils.range_conversion import range_gate_to_range

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    from typing_extensions import Unpack
else:
    from typing import Unpack

TeLibCfg = TypeVar("TeLibCfg", bound=TargetEstimationCfgParams)
TeLibPro = TypeVar("TeLibPro", bound=ExtendedTargetEstimationProParams)


class TargetEstimationProcess(
    Process[
        TargetEstimationCfgParams,
        TargetEstimationProParams,
        MFVariables,
        MFOutArgs,
        AnalysisLib[TeLibCfg, TeLibPro, MFVariables],
    ],
    Generic[TeLibCfg, TeLibPro],
):
    def __init__(
        self,
        cfg_raw: str | Path | TeLibCfg,
        exp_params: ExpDef,
        cfg_params: CfgParams,
        pro_params: ProParams,
        epoch_bounds: Bounds,
        func_get_data: ExtractSignals,
        func_get_pointing: Callable[[int], Pointing],
        output_dir: str | Path | None = None,
        progress: bool = False,
        **kwargs: Unpack[ArrayKwargs],
    ) -> None:
        super().__init__(
            cfg_raw,
            exp_params,
            cfg_params,
            pro_params,
            epoch_bounds,
            func_get_data,
            func_get_pointing,
            output_dir,
            progress,
            **kwargs,
        )
        if isinstance(cfg_raw, TargetEstimationCfgParams):
            self.cfg_params: TeLibCfg = cfg_raw
        elif not isinstance(cfg_raw, dict):
            self.cfg_params = self.get_lib_specific_conf_params(Path(cfg_raw), self.cfg_params)
        self.pro_params: TeLibPro = self.get_lib_specific_process_params(
            exp_params, self.cfg_params, self.pro_params
        )

    def get_conf_params(self, cfg_path: Path, cfg_params: CfgParams) -> TargetEstimationCfgParams:
        """
        Extract Target Estimation parameters

        Args:
            cfg_path: Path to configuration file.
            cfg_params: already loaded configuraion parameters that can be extended

        Returns:
            Process specific Optimize Configuration parameters
        """

        d = extract_config_section(
            cfg_path, ConfigSubSection.TARGET_ESTIMATION, TargetEstimationCfgParams, cfg_params
        )

        return TargetEstimationCfgParams(**d)

    @abstractmethod
    def get_lib_specific_conf_params(self, cfg_path: Path, cfg_params: TargetEstimationCfgParams) -> TeLibCfg:
        pass

    def get_process_params(
        self, exp_params: ExpDef, cfg_params: TargetEstimationCfgParams, pro_params: ProParams
    ) -> TargetEstimationProParams:
        """
        Calculate Optimize specific process parameters

        Args:
            exp_params: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific Optimize Process parameters
        """

        def usec_to_samp(usec: int | float) -> int:
            return int(usec / exp_params.t_samp_usec)

        rx_start_samp = usec_to_samp(exp_params.t_rx_start_usec)
        rx_end_samp = usec_to_samp(exp_params.t_rx_end_usec)
        tx_start_samp = usec_to_samp(exp_params.t_tx_start_usec)
        tx_end_samp = usec_to_samp(exp_params.t_tx_end_usec)

        decimated_read_length = np.ceil(pro_params.read_length / cfg_params.frequency_decimation).astype(
            np.int64
        )

        full_res_rgs = np.arange(
            pro_params.range_gates[0],
            pro_params.range_gates[-1] + cfg_params.range_gate_step,
            cfg_params.range_gate_step / cfg_params.range_gate_sub_resolution,
            dtype=np.float64,
        )

        ranges = range_gate_to_range(full_res_rgs + 1, exp_params.sample_rate).astype(np.float64)  # m

        assert np.all(pro_params.range_gates >= 0), "Computed range gates not compatible with stencils"

        _tx_pulse_samps = tx_end_samp - tx_start_samp

        il0_rgs, il0_rx_window_indices, il1_rx_window_indices = get_ilx_windows(
            exp_params, cfg_params, pro_params
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
        assert exp_params.ipp_samps % cfg_params.frequency_decimation == 0, (
            "ipp-samples length not compatible with decimation: "
            f"{exp_params.ipp_samps} % {cfg_params.frequency_decimation} = "
            f"{exp_params.ipp_samps % cfg_params.frequency_decimation}"
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
        _fft_frequencies = fft.fftfreq(
            decimated_read_length,
            d=cfg_params.frequency_decimation / exp_params.sample_rate,
        )  # Hz

        range_rates = (exp_params.wavelength * _fft_frequencies).astype(np.float64)

        return TargetEstimationProParams(
            **asdict(pro_params),
            decimated_read_length=decimated_read_length,
            il0_rgs=il0_rgs,
            ranges=ranges,
            il1_rx_window_indices=il1_rx_window_indices,
            il0_rx_window_indices=il0_rx_window_indices,
            il0_dec_rx_window_indices=il0_dec_rx_window_indices,
            range_rates=range_rates,
        )

    @abstractmethod
    def get_lib_specific_process_params(
        self,
        exp_params: ExpDef,
        cfg_params: TeLibCfg,
        pro_params: TargetEstimationProParams,
    ) -> TeLibPro:
        pass

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
        exp_params: ExpDef,
        cfg_params: TargetEstimationCfgParams,
        pro_params: TargetEstimationProParams,
    ) -> MFOutArgs:
        """
        Calculate important parameters from the GMF analysis and generate the output.

        Args:
            all_vars: All cohints analysed data stacked together
            file_idx_sample: File id, microseconds since epoch.
            exp_params: Experiment parameters
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
        snr = (np.sqrt(all_vars.vals) - np.sqrt(noise_floor[None, :])) ** 2 / noise_floor[None, :]
        # finding peaks
        r_inds = np.argmax(snr, axis=1)
        r_vec = pro_params.ranges[r_inds]
        v_vec = pro_params.range_rates[all_vars.v_ind[coh_ints, r_inds]]
        a_vec = pro_params.accelerations[all_vars.a_ind[coh_ints, r_inds]]  # type: ignore[attr-defined]
        g_vec = all_vars.vals[coh_ints, r_inds]

        # TODO: Should we save it in usec instead to keep it consistent
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
            ranges=pro_params.ranges,
            range_rates=pro_params.range_rates,
            accelerations=pro_params.accelerations,  # type: ignore[attr-defined]
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
                dims=[(str_dims_num_cohints_per_file, "2")],
                long_name="Radar pointing data (azimuth, elevation)",
            ),
            f"{output.epoch=}".split("=")[0].split(".")[1]: DataItem(
                data=output.epoch,
                long_name="epoch",  # TODO: better description
                scale=True,
            ),
        }
