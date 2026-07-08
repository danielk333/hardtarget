"""Direction of Arrival Process"""

import sys
from dataclasses import asdict

import numpy as np
from pyant.models.array import Array

from hardtarget.constants import AnalysisMethod, ConfigSubSection, DOAMethod, Impl
from hardtarget.interferometry import get_doa_lib
from hardtarget.interferometry.types import DOACfgParams, DOAOutArgs, DOAProParams, DOAVars
from hardtarget.process import Process
from hardtarget.types import (
    DataItem,
    ExpDef,
    InterferometryLib,
    MethodLib,
    ProParams,
)

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    pass
else:
    pass


class DOAProcess(Process[DOACfgParams, DOAProParams, DOAVars, DOAOutArgs, InterferometryLib]):
    method = AnalysisMethod.direction_of_arrival
    config_section = ConfigSubSection.INTERFEROMETRY

    def __post_init__(self) -> None:
        """Extract beam and beam parameters"""

        if "beam" in self.kwargs and "parameters" in self.kwargs:
            self.beam = self.kwargs["beam"]
            self.beam_parameters = self.kwargs["parameters"]
        else:
            raise ValueError("Not possible to run the interferometry calculations without beam data")

        if not isinstance(self.beam, Array):
            raise ValueError("Beam is not an array, not possible to run interferometery")

    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[InterferometryLib, DOAMethod, Impl]:
        return get_doa_lib(lib, impl)

    def get_process_params(
        self, exp_def: ExpDef, cfg_params: DOACfgParams, pro_params: ProParams
    ) -> DOAProParams:
        """
        Calculate interferometry specific process parameters

        Args:
            exp_def: Experiment parameters from measurement file
            cfg_params: Process specific configuration paramters
            pro_params: General process parameters

        Returns:
            Process specific interferometry Process parameters
        """

        kx, ky = np.meshgrid(
            np.linspace(-1, 1, cfg_params.resolution),
            np.linspace(-1, 1, cfg_params.resolution),
        )
        with np.errstate(invalid="ignore"):
            kz = np.sqrt(1 - np.square(kx) - np.square(ky))

        k_1d_index = np.linspace(0, cfg_params.resolution - 1, cfg_params.resolution, dtype=int)
        x_ind, y_ind = np.meshgrid(k_1d_index, k_1d_index)
        k_2d_index = np.rec.fromarrays((y_ind, x_ind))

        return DOAProParams(**asdict(pro_params), kx=kx, ky=ky, kz=kz, k_index=k_2d_index)

    def analyse_ipps(self, start_sample: int) -> DOAVars:
        """
        Analyse the interpulse periods from start sample with the choosen interferometry method

        Args:
            start_sample: sample index to start analysis at.

        Returns:
            Outcome of interferometry analysis
        """

        tx, rx, ipp = self.get_data(
            start_sample=start_sample, read_length=self.pro_params.read_length, sum_rx_channels=False
        )

        return self.lib(rx, self.exp_def, self.cfg_params, self.pro_params, self.beam, self.beam_parameters)

    def stack_vars(self, vars_list: list[DOAVars]) -> DOAVars:
        """Stack the results from the analysis"""

        return DOAVars(
            k_vec=np.stack([x.k_vec for x in vars_list], axis=0),
            peak=np.stack([x.peak for x in vars_list], axis=0),
            azimuth=np.stack([x.azimuth for x in vars_list], axis=0),
            elevation=np.stack([x.elevation for x in vars_list], axis=0),
        )

    def generate_output(
        self,
        all_vars: DOAVars,
        file_idx_sample: int,
        exp_def: ExpDef,
        cfg_params: DOACfgParams,
        pro_params: DOAProParams,
    ) -> DOAOutArgs:
        """
        Restructures the data to a out args object

        Args:
            all_vars: All cohints analysed data stacked together
            file_idx_sample: File id, microseconds since epoch.
            exp_def: Experiment parameters
            cfg_params: Configuration parameters

        Returns:
            Output data
        """

        _t_conv = (cfg_params.n_ipp * exp_def.t_ipp_usec) * 1e-6
        t = (np.arange(len(all_vars.azimuth)) + 1) * _t_conv + file_idx_sample * exp_def.t_samp_usec * 1e-6

        return DOAOutArgs(
            k_vec=all_vars.k_vec,
            peak=all_vars.peak,
            azimuth=all_vars.azimuth,
            elevation=all_vars.elevation,
            epoch_us=int(self.data.epoch_bounds[0]),
            t=t,
        )

    def define_h5_vars(self, output: DOAOutArgs) -> dict[str, DataItem]:
        """
        Appends specifications to the optimize output, such as dimensions, long names, units and more.

        Args:
            output: The output from the analysis

        Returns:
            A dictionary containing the output with attributes such as dimensions, long names and units.
        """

        return {
            f"{output.k_vec=}".split("=")[0].split(".")[1]: DataItem(
                data=output.k_vec,
                long_name="x,y,z for the optimal val for each coherent integration",
            ),
            f"{output.peak=}".split("=")[0].split(".")[1]: DataItem(
                data=output.peak,
                long_name="optimal value for each coherent integration",
            ),
            f"{output.azimuth=}".split("=")[0].split(".")[1]: DataItem(
                data=output.azimuth,
                long_name="azimuth for each coherent integration",
            ),
            f"{output.elevation=}".split("=")[0].split(".")[1]: DataItem(
                data=output.elevation,
                long_name="elevation for each coherent integration",
            ),
            f"{output.epoch_us=}".split("=")[0].split(".")[1]: DataItem(
                data=output.epoch_us,
                long_name="Epoch of the first data sample of the measurement",
            ),
            f"{output.t=}".split("=")[0].split(".")[1]: DataItem(
                data=output.t,
                long_name="time vector relative to epoch_us",
                scale=True,
            ),
        }
