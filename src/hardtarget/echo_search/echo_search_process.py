"""Event search process"""

from dataclasses import asdict

import numpy as np

from hardtarget.constants import AnalysisMethod, ConfigSubSection, EchoSearchMethod, Impl
from hardtarget.echo_search import get_echo_search_lib
from hardtarget.echo_search.types import (
    EchoSearchCfgParams,
    EchoSearchOutArgs,
    EchoSearchProParams,
    EchoSearchVars,
)
from hardtarget.process import Process
from hardtarget.types import DataItem, EventSearchLib, ExpDef, MethodLib, ProParams


class EchoSearchProcess(
    Process[EchoSearchCfgParams, EchoSearchProParams, EchoSearchVars, EchoSearchOutArgs, EventSearchLib]
):
    method = AnalysisMethod.echo_search
    config_section = ConfigSubSection.ECHO_SEARCH

    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[EventSearchLib, EchoSearchMethod, Impl]:
        return get_echo_search_lib(lib, impl)

    def get_process_params(
        self, exp_def: ExpDef, cfg_params: EchoSearchCfgParams, pro_params: ProParams
    ) -> EchoSearchProParams:
        """
        Calculate Optimize specific process parameters

        Args:
            exp_def: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific Optimize Process parameters
        """

        doppler_frequencies = np.arange(
            cfg_params.doppler_freq_min,
            cfg_params.doppler_freq_max + cfg_params.doppler_freq_step,
            cfg_params.doppler_freq_step,
            dtype=np.int32,
        )

        return EchoSearchProParams(**asdict(pro_params), doppler_frequencies=doppler_frequencies)

    def analyse_ipps(self, start_sample: int) -> EchoSearchVars:
        """
        Analyse the interpulse periods from start sample with the choosen optimize method

        Args:
            start_sample: sample index to start analysis at.

        Returns:
            Outcome of xcorr analysis
        """

        tx, rx, ipp = self.get_data(start_sample, self.pro_params.read_length, sum_rx_channels=False)

        return self.lib(
            tx,
            rx,
            self.exp_def,
            self.cfg_params,
            self.pro_params,
        )

    def stack_vars(self, vars_list: list[EchoSearchVars]) -> EchoSearchVars:
        """Stack the results from the analysis"""

        return EchoSearchVars(
            max_corr=np.stack([x.max_corr for x in vars_list], axis=0),
            max_corr_ind=np.stack([x.max_corr_ind for x in vars_list], axis=0),
            max_corr_delay=np.stack([x.max_corr_delay for x in vars_list], axis=0),
            best_doppler=np.stack([x.best_doppler for x in vars_list], axis=0),
            tot_pow=np.stack([x.tot_pow for x in vars_list], axis=0),
            mean=np.stack([x.mean for x in vars_list], axis=0),
            std_dev=np.stack([x.std_dev for x in vars_list], axis=0),
        )

    def generate_output(
        self,
        all_vars: EchoSearchVars,
        file_idx_sample: int,
        exp_def: ExpDef,
        cfg_params: EchoSearchCfgParams,
        pro_params: EchoSearchProParams,
    ) -> EchoSearchOutArgs:
        """
        In this case does nothing as the data is already in the correct format

         Args:
             all_vars: All cohints analysed data stacked together
             file_idx_sample: File id, sample relative to file start.
             exp_def: Experiment parameters
             cfg_params: Configuration parameters

         Returns:
             Output data
        """

        epoch_us = int(self.data.epoch_bounds[0] + file_idx_sample * exp_def.t_samp_usec)

        return EchoSearchOutArgs(
            max_corr=all_vars.max_corr
            if isinstance(all_vars.max_corr, np.ndarray)
            else np.array(all_vars.max_corr),
            max_corr_ind=all_vars.max_corr_ind
            if isinstance(all_vars.max_corr_ind, np.ndarray)
            else np.array(all_vars.max_corr_ind),
            max_corr_delay=all_vars.max_corr_delay
            if isinstance(all_vars.max_corr_delay, np.ndarray)
            else np.array(all_vars.max_corr_delay),
            best_doppler=all_vars.best_doppler
            if isinstance(all_vars.best_doppler, np.ndarray)
            else np.array(all_vars.best_doppler),
            tot_pow=all_vars.tot_pow
            if isinstance(all_vars.tot_pow, np.ndarray)
            else np.array(all_vars.tot_pow),
            mean=all_vars.mean if isinstance(all_vars.mean, np.ndarray) else np.array(all_vars.mean),
            std_dev=all_vars.std_dev
            if isinstance(all_vars.std_dev, np.ndarray)
            else np.array(all_vars.std_dev),
            epoch_us=epoch_us,
        )

    def define_h5_vars(self, output: EchoSearchOutArgs) -> dict[str, DataItem]:
        """
        Appends specifications to the optimize output, such as dimensions, long names, units and more.

        Args:
            output: The output from the analysis

        Returns:
            A dictionary containing the output with attributes such as dimensions, long names and units.
        """
        return {
            f"{output.max_corr=}".split("=")[0].split(".")[1]: DataItem(
                data=output.max_corr,
                long_name=" Best correlation from all doppler frequencies.",
            ),
            f"{output.max_corr_ind=}".split("=")[0].split(".")[1]: DataItem(
                data=output.max_corr_ind,
                long_name="Index of the max_corr within the correlation array for the best doppler.",
            ),
            f"{output.max_corr_delay=}".split("=")[0].split(".")[1]: DataItem(
                data=output.max_corr_delay,
                long_name="Delay of the max_corr_ind within the correlation array for the best doppler.",
            ),
            f"{output.best_doppler=}".split("=")[0].split(".")[1]: DataItem(
                data=output.best_doppler,
                long_name="Doppler frequency that contained the best correlation.",
            ),
            f"{output.tot_pow=}".split("=")[0].split(".")[1]: DataItem(
                data=output.tot_pow,
                long_name="The total power of the IPP",
            ),
            f"{output.mean=}".split("=")[0].split(".")[1]: DataItem(
                data=output.mean,
                long_name="Mean power per ipp",
                # shape: ipps, channels
            ),
            f"{output.std_dev=}".split("=")[0].split(".")[1]: DataItem(
                data=output.std_dev,
                long_name="Standard deviation of power per ipp",
                # shape: ipps, channels
            ),
            f"{output.epoch_us=}".split("=")[0].split(".")[1]: DataItem(
                data=output.epoch_us,
                long_name="Epoch of the first analysed datapoint in microseconds",
            ),
        }
