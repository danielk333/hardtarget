"""General Matched Filter Optimization Process"""

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from radardef.components import DataLoader

from hardtarget.constants import AnalysisMethod, ConfigSubSection, Impl, OptimizationMethod
from hardtarget.optimization import get_optimize_lib
from hardtarget.optimization.types import (
    MFOptimizeOutArgs,
    MFOptimizeVariables,
    OptimizeCfgParams,
    OptimizeProParams,
    OptStart,
)
from hardtarget.process import Process
from hardtarget.process.configuration import extract_config_section, get_ilx_windows
from hardtarget.target_estimation.gmf import GMFCfgParams
from hardtarget.target_estimation.types import MFOutArgs
from hardtarget.types import (
    CfgParams,
    DataItem,
    ExpDef,
    MethodLib,
    OptimizeLib,
    ProParams,
)
from hardtarget.utils.h5_tools import get_analysed_h5_files


class OptimizeProcess(
    Process[OptimizeCfgParams, OptimizeProParams, MFOptimizeVariables, MFOptimizeOutArgs, OptimizeLib]
):
    method = AnalysisMethod.optimize

    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[OptimizeLib, OptimizationMethod, Impl]:
        return get_optimize_lib(lib, impl)

    def __init__(
        self,
        config: str | Path | OptimizeCfgParams,
        data: DataLoader,
        method_lib: Optional[MethodLib] = None,
        impl: Optional[Impl] = None,
        rx_channel: Optional[str | int] = None,
        excluded_channels: Optional[list[str] | list[int]] = None,
        output_dir: Optional[str | Path] = None,
        progress: bool = False,
    ) -> None:
        super().__init__(
            config,
            data,
            method_lib,
            impl,
            rx_channel,
            excluded_channels,
            output_dir,
            progress,
        )

        estimated_data = Path(self.cfg_params.path).resolve()
        # If output directory is the same as source, add data
        if str(estimated_data.parent) == str(self.output_dir):
            self.store_mode = "a"
            self.store_params = False

        # Extract all already analysed files
        paths = get_analysed_h5_files(estimated_data)
        paths.sort()
        self.sorted_mf_files = paths

        # Verify optimization is running on the correct data
        with h5py.File(self.sorted_mf_files[0], "r") as hf:
            method = AnalysisMethod(hf[f"{ProParams.method=}".split("=")[0].split(".")[1]].asstr()[()])
            if method != AnalysisMethod.target_estimation:
                raise ValueError(
                    f"It is only possible to run optimization on a previous target estimation analysis, not on {method}"
                )
            try:
                self.sub_resolution: int = hf[GMFCfgParams.__name__][
                    f"{GMFCfgParams.range_gate_sub_resolution=}".split("=")[0].split(".")[1]
                ][()]
                # Extract sample relative to file start
                self.mf_sample_start: int = int(
                    (
                        self.data.epoch_bounds[0]
                        - int(hf["OutArgs"][f"{MFOutArgs.epoch_us=}".split("=")[0].split(".")[1]][()])
                    )
                    / self.exp_params.t_samp_usec
                )
            except KeyError:
                raise ValueError("Optimization is only compatible with a previous gmf analysis")

    def load_analysed_cohint(self, start_sample: int) -> OptStart:
        """
        Extract range, velocity and acceleration estimation from a specific cohint based on the start_sample
        from a previous analysis

        Args:
            start_sample: start sample to be analysed, cohint will be calculated from this
        Returns:
            Range, velocity and acceleration estimation from previous analysis for the specific cohint
        """

        relative_sample_index = start_sample - self.mf_sample_start
        samples_per_cohint = self.exp_params.ipp_samps * self.cfg_params.n_ipp
        samples_per_file = samples_per_cohint * self.cfg_params.num_cohints_per_file
        file_id = relative_sample_index // samples_per_file
        cohind = (start_sample - (file_id * samples_per_file)) // samples_per_cohint

        with h5py.File(self.sorted_mf_files[file_id], "r") as hf:
            # TODO: Should r/v/a_vec from all files be read at start and loaded to RAM to access it faster?
            opt_start = OptStart(
                r_vec=hf["OutArgs"][f"{MFOutArgs.r_vec=}".split("=")[0].split(".")[1]][cohind],
                v_vec=hf["OutArgs"][f"{MFOutArgs.v_vec=}".split("=")[0].split(".")[1]][cohind],
                a_vec=hf["OutArgs"][f"{MFOutArgs.a_vec=}".split("=")[0].split(".")[1]][cohind],
                dc=hf["OutArgs"][f"{MFOutArgs.dc=}".split("=")[0].split(".")[1]][cohind],
                t=hf["OutArgs"][f"{MFOutArgs.t=}".split("=")[0].split(".")[1]][cohind],
            )

        return opt_start

    def get_conf_params(self, cfg_path: Path, cfg_params: CfgParams) -> OptimizeCfgParams:
        """
        Extract Optimize configuration parameters

        Args:
            cfg_path: Path to configuration file.
            cfg_params: already loaded configuraion parameters that can be extended

        Returns:
            Process specific Optimize Configuration parameters
        """

        d = extract_config_section(
            cfg_path,
            ConfigSubSection.OPTIMIZATION,
            OptimizeCfgParams,
            cfg_params,
            self._logger,
        )

        return OptimizeCfgParams(**d)

    def get_process_params(
        self, exp_params: ExpDef, cfg_params: OptimizeCfgParams, pro_params: ProParams
    ) -> OptimizeProParams:
        """
                Calculate Optimize specific process parameters

        Args:
            exp_params: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters
        Returns:
            Process specific Optimize Process parameters
        """

        _, il0_rx_window_indices, _ = get_ilx_windows(exp_params, cfg_params, pro_params)

        return OptimizeProParams(**asdict(pro_params), il0_rx_window_indices=il0_rx_window_indices)

    def analyse_ipps(self, start_sample: int) -> MFOptimizeVariables:
        """
        Analyse the interpulse periods from start sample with the choosen optimize method

        Args:
            start_sample: sample index to start analysis at.

        Returns:
            Outcome of optimize analysis
        """

        tx, rx, ipp = self.get_data(
            start_sample, self.pro_params.read_length, sub_resolution=self.sub_resolution
        )

        # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
        # scale transmit waveform to unity power
        tx_pwr = np.sum(np.abs(tx) ** 2.0)
        tx_amp = np.sqrt(tx_pwr)
        tx = np.conj(tx) / tx_amp

        opt_start = self.load_analysed_cohint(start_sample)

        gmf_vars = MFOptimizeVariables(
            r_vec_opt=np.array([opt_start.r_vec], dtype=np.float64),
            v_vec_opt=np.array([opt_start.r_vec], dtype=np.float64),
            a_vec_opt=np.array([opt_start.r_vec], dtype=np.float64),
            peak_vals=np.zeros((1,), dtype=np.float64),
            dc=np.array([opt_start.dc]),
            t=np.array([opt_start.r_vec]),
        )

        if tx_amp > self.cfg_params.tx_amp_limit:
            r_vec_opt, v_vec_opt, a_vec_opt, peak_vals = self.lib(
                tx[:, 0],  # TODO: How to handle sub resolution
                ipp,
                self.exp_params,
                self.cfg_params,
                self.pro_params,
                opt_start.r_vec,
                opt_start.v_vec,
                opt_start.a_vec,
            )
            return MFOptimizeVariables(
                r_vec_opt=np.asarray(r_vec_opt),
                v_vec_opt=np.asarray(v_vec_opt),
                a_vec_opt=np.asarray(a_vec_opt),
                peak_vals=np.asarray(peak_vals),
                dc=np.asarray(opt_start.dc),
                t=np.asarray(opt_start.t),
            )

        return gmf_vars

    def stack_vars(self, vars_list: list[MFOptimizeVariables]) -> MFOptimizeVariables:
        """Stack the results from the analysis"""

        return MFOptimizeVariables(
            r_vec_opt=np.stack([x.r_vec_opt for x in vars_list], axis=0),
            v_vec_opt=np.stack([x.v_vec_opt for x in vars_list], axis=0),
            a_vec_opt=np.stack([x.a_vec_opt for x in vars_list], axis=0),
            peak_vals=np.stack([x.peak_vals for x in vars_list], axis=0),
            dc=np.stack([x.dc for x in vars_list], axis=0),
            t=np.stack([x.t for x in vars_list], axis=0),
        )

    def generate_output(
        self,
        all_vars: MFOptimizeVariables,
        file_idx_sample: int,
        exp_params: ExpDef,
        cfg_params: OptimizeCfgParams,
        pro_params: OptimizeProParams,
    ) -> MFOptimizeOutArgs:
        """
        Restructures the data to a out args object

        Args:
            all_vars: All cohints analysed data stacked together
            file_idx_sample: File id, microseconds since epoch.
            exp_params: Experiment parameters
            cfg_params: Configuration parameters

        Returns:
            Output data
        """
        return all_vars

    def define_h5_vars(self, output: MFOptimizeOutArgs) -> dict[str, DataItem]:
        """
        Appends specifications to the optimize output, such as dimensions, long names, units and more.

        Args:
            output: The output from the analysis

        Returns:
            A dictionary containing the output with attributes such as dimensions, long names and units.
        """
        return {
            f"{output.r_vec_opt=}".split("=")[0].split(".")[1]: DataItem(
                data=output.r_vec_opt,
                # dims=[("num_cohints_per_file", "t")], #TODO
                long_name="Fine tuned range",
            ),
            f"{output.v_vec_opt=}".split("=")[0].split(".")[1]: DataItem(
                data=output.v_vec_opt,
                # dims=[("num_cohints_per_file", "t")], #TODO
                long_name="Fine tuned range-rate",
            ),
            f"{output.a_vec_opt=}".split("=")[0].split(".")[1]: DataItem(
                data=output.a_vec_opt,
                # dims=[("num_cohints_per_file", "t")], #TODO
                long_name="Fine tuned acceleration",
            ),
            f"{output.peak_vals=}".split("=")[0].split(".")[1]: DataItem(
                data=output.peak_vals,
                # dims=[("num_cohints_per_file", "t")], #TODO
                long_name="Generalized Matched Filter fine tuned peak output values",
            ),
            f"{output.dc=}".split("=")[0].split(".")[1]: DataItem(
                data=output.dc,
                # dims=[(str_dims_num_cohints_per_file, str_t), (str_ranges, "r")],
                long_name="Range dependant noise floor (0-frequency gmf output)",
            ),
            f"{output.t=}".split("=")[0].split(".")[1]: DataItem(
                data=output.t,
                long_name="time vector",
                scale=True,
            ),
        }
