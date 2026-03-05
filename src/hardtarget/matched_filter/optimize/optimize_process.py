"""General Matched Filter Optimization Process"""

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from hardtarget.data_handling.configuration import extract_config_section
from hardtarget.matched_filter.optimize import get_optimize_lib
from hardtarget.matched_filter.optimize.types import OptimizeCfgParams, OptimizeProParams
from hardtarget.matched_filter.types import MFOptimizeOutArgs, MFOptimizeVariables
from hardtarget.process import Process
from hardtarget.types.constants import ConfigSubSection, Impl, OptimizationMethod
from hardtarget.types.types import (
    Bounds,
    CfgParams,
    DataItem,
    ExpParams,
    ExtractedSignals,
    MethodLib,
    OptimizeLib,
    OptStart,
    Pointing,
    ProParams,
)
from hardtarget.utils.h5_tools import get_analysed_h5_files


class OptimizeProcess(
    Process[OptimizeCfgParams, OptimizeProParams, MFOptimizeVariables, MFOptimizeOutArgs, OptimizeLib]
):
    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[OptimizeLib, OptimizationMethod, Impl]:
        return get_optimize_lib(lib, impl)

    def __init__(
        self,
        cfg_path: Path,
        exp_params: ExpParams,
        cfg_params: CfgParams,
        pro_params: ProParams,
        epoch_bounds: Bounds,
        func_get_data: Callable[[int, int], ExtractedSignals],
        func_get_pointing: Callable[[int], Pointing],
        output_dir: str | Path | None = None,
        progress: bool = False,
    ) -> None:
        super().__init__(
            cfg_path,
            exp_params,
            cfg_params,
            pro_params,
            epoch_bounds,
            func_get_data,
            func_get_pointing,
            output_dir,
            progress,
        )

        # Extract all already analysed files
        paths = get_analysed_h5_files(self.cfg_params.path)
        paths.sort()
        self.sorted_mf_files = paths

        # Override default params, add data to existing file do not save params again
        self.store_mode = "a"
        self.store_params = False

    def load_analysed_cohint(self, start_sample: int) -> OptStart:
        """
        Extract range, velocity and acceleration estimation from a specific cohint based on the start_sample
        from a previous analysis

        Args:
            start_sample: start sample to be analysed, cohint will be calculated from this
        Returns:
            Range, velocity and acceleration estimation from previous analysis for the specific cohint
        """

        relative_sample_index = start_sample - self.sample_bounds.start
        samples_per_cohint = self.exp_params.ipp_samps * self.cfg_params.n_ipp
        samples_per_file = samples_per_cohint * self.cfg_params.num_cohints_per_file
        file_id = relative_sample_index // samples_per_file
        cohind = (start_sample - (file_id * samples_per_file)) // samples_per_cohint

        with h5py.File(self.sorted_mf_files[file_id], "r") as hf:
            group = hf["OutArgs"]  # TODO: Update to proper type
            # TODO: Should r/v/a_vec from all files be read at start and loaded to RAM to access it faster?
            # TODO hardcode the r_vec string
            opt_start = OptStart(
                r_vec=group["r_vec"][cohind], v_vec=group["v_vec"][cohind], a_vec=group["a_vec"][cohind]
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
        self, exp_params: ExpParams, cfg_params: OptimizeCfgParams, pro_params: ProParams
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

        return OptimizeProParams(**asdict(pro_params))

    def analyse_ipps(self, start_sample: int) -> MFOptimizeVariables:
        """
        Analyse the interpulse periods from start sample with the choosen optimize method

        Args:
            start_sample: sample to start the analysis

        Returns:
            Outcome of optimize analysis
        """

        tx, rx, ipp = self.get_data(start_sample, self.pro_params.read_length)

        # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
        # scale transmit waveform to unity power
        tx_pwr = np.sum(np.abs(tx) ** 2.0)
        tx_amp = np.sqrt(tx_pwr)
        tx = np.conj(tx) / tx_amp

        gmf_vars = MFOptimizeVariables(
            peak=np.zeros((3,), dtype=np.float64),  # TODO: Correct size
            peak_val=np.zeros((1,), dtype=np.float64),
        )

        if tx_amp > self.cfg_params.tx_amp_limit:
            gmf_vars.peak[:], gmf_vars.peak_val[0] = self.lib(
                tx,
                ipp,
                self.exp_params,
                self.cfg_params,
                self.pro_params,
                self.load_analysed_cohint(start_sample),
            )

        return gmf_vars

    def stack_vars(self, vars_list: list[MFOptimizeVariables]) -> MFOptimizeVariables:
        """Stack the results from the analysis"""

        return MFOptimizeVariables(
            peak=np.stack([x.peak for x in vars_list], axis=0),
            peak_val=np.stack([x.peak_val for x in vars_list], axis=0),
        )

    def generate_output(
        self,
        all_vars: MFOptimizeVariables,
        file_idx_sample: int,
        exp_params: ExpParams,
        cfg_params: OptimizeCfgParams,
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

        return MFOptimizeOutArgs(
            peaks=all_vars.peak,
            peak_vals=all_vars.peak_val,
        )

    def define_h5_vars(self, output: MFOptimizeOutArgs) -> dict[str, DataItem]:
        """
        Appends specifications to the optimize output, such as dimensions, long names, units and more.

        Args:
            output: The output from the analysis

        Returns:
            A dictionary containing the output with attributes such as dimensions, long names and units.
        """
        return {
            f"{output.peaks=}".split("=")[0].split(".")[1]: DataItem(  # gmf_optimized_peak
                data=output.peaks,
                # dims=[("num_cohints_per_file", "t")], #TODO
                long_name="Fine tuned range, range-rate and acceleration",
            ),
            f"{output.peak_vals=}".split("=")[0].split(".")[1]: DataItem(  # gmf_optimized
                data=output.peak_vals,
                # dims=[("num_cohints_per_file", "t")], #TODO
                long_name="Generalized Matched Filter fine tuned peak output values",
            ),
        }
