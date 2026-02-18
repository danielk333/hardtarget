"""Event search process"""

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hardtarget.data_handling.configuration import extract_config_section
from hardtarget.matched_filter.xcorr import get_event_detection_lib
from hardtarget.matched_filter.xcorr.types import (
    XCorrCfgParams,
    XCorrProParams,
    XCorrVariables,
)
from hardtarget.process import Process
from hardtarget.types.constants import ConfigSubSection, EventDetectionMethod, Impl
from hardtarget.types.types import CfgParams, DataItem, EventSearchLib, ExpParams, MethodLib, ProParams


class XCorrProcess(Process[XCorrCfgParams, XCorrProParams, XCorrVariables, XCorrVariables, EventSearchLib]):
    logger = logging.getLogger(__name__)

    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[EventSearchLib, EventDetectionMethod, Impl]:
        return get_event_detection_lib(lib, impl)

    def get_conf_params(self, cfg_path: Path, cfg_params: CfgParams) -> XCorrCfgParams:
        """
        Extract Optimize configuration parameters

        Args:
            cfg_path: Path to configuration file.
            cfg_params: already loaded configuraion parameters that can be extended

        Returns:
            Process specific Optimize Configuration parameters
        """

        d = extract_config_section(cfg_path, ConfigSubSection.XCORR, XCorrCfgParams, cfg_params)

        # TODO: if n_ipp is above 1 send a warning to the user and then continue, not sure if works with many

        return XCorrCfgParams(**d)

    def get_process_params(
        self, exp_params: ExpParams, cfg_params: XCorrCfgParams, pro_params: ProParams
    ) -> XCorrProParams:
        """
        Calculate Optimize specific process parameters

        Args:
            exp_params: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific Optimize Process parameters
        """

        doppler_freq_size = int(
            ((cfg_params.doppler_freq_max - cfg_params.doppler_freq_min) / cfg_params.doppler_freq_step) + 1
        )

        return XCorrProParams(**asdict(pro_params), doppler_freq_size=doppler_freq_size)

    def analyse_ipps(self, start_sample: int) -> XCorrVariables:
        """
        Analyse the interpulse periods from start sample with the choosen optimize method

        Args:
            start_sample: sample index to start analysis at.

        Returns:
            Outcome of xcorr analysis
        """

        tx, rx, ipp = self.get_data(start_sample, self.pro_params.read_length)

        return self.lib(tx, rx, self.exp_params, self.cfg_params, self.pro_params)

    def stack_vars(self, vars_list: list[XCorrVariables]) -> XCorrVariables:
        """Stack the results from the analysis"""

        return XCorrVariables(
            max_pow=np.stack([x.max_pow for x in vars_list], axis=0),
            max_pow_norm=np.stack([x.max_pow_norm for x in vars_list], axis=0),
            max_peak=np.stack([x.max_peak for x in vars_list], axis=0),
            max_pow_ind=np.stack([x.max_pow_ind for x in vars_list], axis=0),
            best_doppler=np.stack([x.best_doppler for x in vars_list], axis=0),
            ipps_pow=np.stack([x.ipps_pow for x in vars_list], axis=0),
        )

    def generate_output(
        self,
        all_vars: XCorrVariables,
        file_idx_sample: int,
        exp_params: ExpParams,
        cfg_params: XCorrCfgParams,
    ) -> XCorrVariables:
        """
        In this case does nothing as the data is already in the correct format

         Args:
             all_vars: All cohints analysed data stacked together
             file_idx_sample: File id, microseconds since epoch.
             exp_params: Experiment parameters
             cfg_params: Configuration parameters

         Returns:
             Output data
        """

        return all_vars

    def define_h5_vars(self, output: XCorrVariables) -> dict[str, DataItem]:
        """
        Appends specifications to the optimize output, such as dimensions, long names, units and more.

        Args:
            output: The output from the analysis

        Returns:
            A dictionary containing the output with attributes such as dimensions, long names and units.
        """
        return {
            f"{output.max_pow=}".split("=")[0].split(".")[1]: DataItem(
                data=output.max_pow,
                long_name="TODO",
            ),
            f"{output.max_pow_norm=}".split("=")[0].split(".")[1]: DataItem(
                data=output.max_pow_norm,
                long_name="TODO",
            ),
            f"{output.max_peak=}".split("=")[0].split(".")[1]: DataItem(
                data=output.max_peak,
                long_name="TODO",
            ),
            f"{output.max_pow_ind=}".split("=")[0].split(".")[1]: DataItem(
                data=output.max_pow_ind,
                long_name="TODO",
            ),
            f"{output.best_doppler=}".split("=")[0].split(".")[1]: DataItem(
                data=output.best_doppler,
                long_name="TODO",
            ),
            f"{output.ipps_pow=}".split("=")[0].split(".")[1]: DataItem(
                data=output.ipps_pow,
                long_name="Total power for each ipp",
            ),
        }
