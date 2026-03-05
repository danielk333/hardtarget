"""
Process, the class represents an arbitrary radar data analysis process. This is a skeleton for the process
and controls the general flow. To create a complete process the class needs to be inherited and the
abstract methods filled in. It supports a variety of datatypes.
"""

import datetime as dt
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Generic, Optional

import numpy as np
from tqdm import tqdm

import hardtarget.process.utils as utils
from hardtarget.data_handling import dump_params_to_file
from hardtarget.process.utils import calculate_tasks, sample_interval_to_closest_ipp
from hardtarget.types.constants import Impl, MethodLib
from hardtarget.types.types import (
    AnalysedResult,
    Bounds,
    CfgParams,
    DataItem,
    ExpParams,
    ExtractedSignals,
    GenericCfg,
    GenericLib,
    GenericOut,
    GenericPro,
    GenericVars,
    Job,
    Pointing,
    ProParams,
)
from hardtarget.utils.time_conversion import time_interval_to_sample_bound, ts_from_str


class Process(ABC, Generic[GenericCfg, GenericPro, GenericVars, GenericOut, GenericLib]):
    """
    Process chain for analysing radar data, configurable based on user configuration parameters.

    The process chain supports multiple types to be able to support a multiple kinds of analysis:

        - GenericCfg: Configurable parameters, process specific section (bound to CfgPro)
        - GenericPro: Process specific parameters deriver from Cfg (bound to ProParams)
        - GenericVars: Datatype produced by the analysis
        - GenericOut: Processed Vars data
        - GenericLib: Library type used for the analysis

    The user configurable parameters are derived during class initialisation:

        1. Get process specific configuration parameters <ref get_conf_params>[type: *GenericCfg*]
            (number of coherent integrations per file, number of ipps, ...)
        2. Get process specific configuration parameters <ref get_process_params>[type: *GenericPro*]
            (range gates, ...)

    From this the processing can be triggered with *run(..)*, this triggers the chain:

    ```



    3.Calculate tasks - [0,1,..,T]
        |
        └>Split tasks for parallelization - N processes
            |                           |
            └>For task [0,..,T-N]       └>For task [1,..,T-(N-1)]  ......
                |                           |
                └>Calculate start sample    └.......
                    |                               :
                    └>For each coherent integration └.....
                    |   |                                 :
                    |   └>Analyse interval <analyse_ipps(..)> [Type: GenericLib]
                    |       |                             :
                    |       └>Store analysed data [Type: GenericVars]
                    |                                     :
                    └>Gather all data <stack_vars(GenericVars)> [Type: GenericVars]
                            |                             :
                            └>Calculate further parameters from the analysis data <generate_output(GenericVars)> [Type: GenericOut]
                                |                         :
                                └>Add attributes <define_h5_vars(GenericOut)>
                                    |                     :
                                    └---------------------└---> Save Data <save_task_data()>

    ```
    Files are stored in the same folder, one file per task.


    Args:
        cfg_path: path to user config file
        exp_params: experiment parameters derived from the radar data
        cfg_params: base configurable processing params
        pro_params: base calculated processing params
        epoch_bounds: time bounds of the data
        func_get_data: function that extracts N rx and tx samples from a given start sample
        pointing_data: radar beam pointing data
        lib: The analysis library
        output_dir: Path to output directory

    """

    def __init__(
        self,
        cfg_raw: str | Path | GenericCfg,
        exp_params: ExpParams,
        cfg_params: CfgParams,
        pro_params: ProParams,
        epoch_bounds: Bounds,
        func_get_data: Callable[[int, int], ExtractedSignals],
        func_get_pointing: Callable[[int], Pointing],
        output_dir: Optional[str | Path] = None,
        progress: bool = False,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self.exp_params = exp_params
        if isinstance(cfg_raw, CfgParams):
            self.cfg_params = cfg_raw
        else:
            self.cfg_params = self.get_conf_params(Path(cfg_raw), cfg_params)
        self.lib, lib_name, impl = self.get_analysis_lib(pro_params.method_lib, pro_params.implementation)
        pro_params = self.define_method_lib(pro_params, lib_name, impl)
        self.pro_params = self.get_process_params(self.exp_params, self.cfg_params, pro_params)
        self.epoch = epoch_bounds
        self.get_data = func_get_data
        self.progress = progress
        self.get_pointing = func_get_pointing
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.progress_bar = None
        self.store_mode = "w"
        self.store_params = True

    @abstractmethod
    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[GenericLib, MethodLib, Impl]:
        pass

    @abstractmethod
    def get_conf_params(self, cfg_path: Path, cfg_params: CfgParams) -> GenericCfg:
        """Abstract method, process specific configuration parameters"""
        pass

    def define_method_lib(self, pro_params: ProParams, lib_name: MethodLib, impl: Impl) -> ProParams:
        """
        If method lib and implementation is not defined or not matching with what is declared in the process
        parameters, correct process parameters and return.
        """

        method_lib_correction = pro_params.method_lib is None or pro_params.method_lib is not lib_name
        implementation_correction = pro_params.implementation is None or pro_params.implementation is not impl

        if method_lib_correction or implementation_correction:
            pro_dict = asdict(pro_params)
            pro_dict[f"{ProParams.method_lib=}".split("=")[0].split(".")[1]] = lib_name
            pro_dict[f"{ProParams.implementation=}".split("=")[0].split(".")[1]] = impl
            return ProParams(**pro_dict)
        else:
            return pro_params

    @abstractmethod
    def get_process_params(
        self, exp_params: ExpParams, cfg_params: GenericCfg, pro_params: ProParams
    ) -> GenericPro:
        """Abstract method, process specific parameters"""
        pass

    @abstractmethod
    def analyse_ipps(self, start_sample: int) -> GenericVars:
        """
        Abstract method, shall analyse the interpulse periods from start sample.

        Args:
            start_sample: sample to start the analysis

        Returns:
            Outcome of analysis
        """

        pass

    @abstractmethod
    def stack_vars(self, vars_list: list[GenericVars]) -> GenericVars:
        """Abstract method, shall stack the results from the analysis"""
        pass

    @abstractmethod
    def generate_output(
        self,
        all_vars: GenericVars,
        file_idx_sample: int,
        exp_params: ExpParams,
        cfg_params: GenericCfg,
    ) -> GenericOut:
        """
        Abstract method, shall calculate important parameters from the analysis and generate the output.

        Args:
            all_vars: All cohints analysed data stacked together
            file_idx_sample: File id, microseconds since epoch.
            exp_params: Experiment parameters
            cfg_params: Configuration parameters

        Returns:
            Output data
        """

        pass

    @abstractmethod
    def define_h5_vars(self, output: GenericOut) -> dict[str, DataItem]:
        """
        Abstract method, shall add attributes to the data.

        Args:
            output: The output from the analysis

        Returns:
            A dictionary containing the output with attributes such as dimensions, long names and units.
        """
        pass

    def process_task(self, task_idx: int, file_idx_sample: int, bounds: Bounds) -> GenericOut:
        """
        Process one task, extract amount of samples to process, analyse the samples for each coherent
        integration, gather results and generate the output result.

        Args:
            task_idx: Task id
            file_idx_sample: file id sample
            bounds: Bounds

        Returns
            Out data suitable for the specific process
        """

        ipp = self.exp_params.t_ipp_usec
        sample_rate = self.exp_params.sample_rate
        n_ipp = self.cfg_params.n_ipp
        num_cohints_per_file = self.cfg_params.num_cohints_per_file
        ipp_samp = self.exp_params.ipp_samps

        ts0 = time.time()

        # --- Make sure the data stays within bounds ---
        num_cohints = num_cohints_per_file
        if file_idx_sample + num_cohints_per_file * ipp_samp * n_ipp - 1 > bounds.end:
            num_cohints = int((bounds.end - file_idx_sample) // (ipp_samp * n_ipp))
        start_cohind = 0

        # --- Optimize and gather vars ---
        collected_vars = []
        for coh_ind in range(start_cohind, num_cohints):
            start_sample = file_idx_sample + coh_ind * ipp_samp * n_ipp
            vars = self.analyse_ipps(start_sample)

            if self.progress_bar is not None:
                self.progress_bar.update(1)

            collected_vars.append(vars)

        ts1 = time.time()

        # --- Concatenate vars ---
        all_vars = self.stack_vars(collected_vars)

        info = {
            "task": task_idx,
            "time": ts1 - ts0,
            "real": (ts1 - ts0) / (n_ipp * ipp * 1e-6 / sample_rate),
        }
        msg = "task_idx {task:4} time {time:1.2f} cpu/real {real:1.2f}"
        self._logger.debug(msg.format(**info))

        # --- Generate output ---
        return self.generate_output(all_vars, file_idx_sample, self.exp_params, self.cfg_params)

    def save_task_data(
        self,
        file_idx_sample: int,
        out_data: GenericOut,
        results: AnalysedResult,
        filepath: Optional[Path] = None,
        clobber: bool = True,
    ) -> AnalysedResult:
        """
        Append attributes to the Out data and save to file

        Args:
            file_idx_sample: file sample
            out_data: analysed output data
            results: results to add data to
            filepath (optional): filepath to store data, if None data will be stored in RAM
            clobber (optional): Overwrite previous datasets, default True
        Returns:
            Analysed result, a collection of the directory and paths with the location of the result. If
            storing the data is unwanted the actual data is stored in the object instead.
        """

        if filepath is not None:
            data = self.define_h5_vars(out_data)

            dump_params_to_file(
                data.items(),
                self.exp_params,
                self.cfg_params,
                self.pro_params,
                filepath,
                clobber=clobber,
                mode=self.store_mode,
                include_params=self.store_params,
            )

            self._logger.debug(f"Analysed data stored in {filepath}")

            results["files"].append(filepath.name)
        else:
            # Write data to dict at file_idx_sample
            results["data"][file_idx_sample] = (out_data, self.exp_params, self.cfg_params, self.pro_params)

        return results

    def run(
        self,
        job: Job,
        channel_bounds: Bounds,
        epoch: Bounds,
        start_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
        end_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
        relative_time: bool = False,
        clobber: bool = True,
    ) -> AnalysedResult:
        """
        Run gathers all components of the process and runs the analysis.

        1. Calculating bounds of data to analyse
        2. Calculate amount of tasks to run
        3. Process tasks
        4. Store data

        Args:
            job: Job id
            channel_bounds: Measurement channel sample bounds
            epoch: Measurement bounds (start and end), in microseconds since epoch.
            start_time (optional): Start time, if set data before this will be neglected
            end_time (optional): End time, if set data after this will be neglected
            relative_time (optional): If relative time should be used
            clobber (optional): Overwrite previous datasets, default True
        Returns:
            Analysed result, a collection of the directory and paths with the location of the result. If
            storing the data is unwanted the actual data is stored in the object instead.
        """

        if isinstance(start_time, str):
            start_time = int(ts_from_str(start_time) * 1e6)
        if isinstance(end_time, str):
            end_time = int(ts_from_str(end_time) * 1e6)

        # bounds
        sample_bounds = time_interval_to_sample_bound(
            start_time=start_time,
            end_time=end_time,
            time_bounds=epoch,
            sample_rate=self.exp_params.sample_rate,
            relative_time=relative_time,
        )

        # round off to closest ipp, starting in the middle of a ipp will cause issues to the analysis
        sample_bounds = sample_interval_to_closest_ipp(
            sample_bounds=sample_bounds, ipp_samps=self.exp_params.ipp_samps
        )

        # add potential sample offset
        self.sample_bounds = Bounds(
            sample_bounds.start + self.cfg_params.samp_offset, sample_bounds.end + self.cfg_params.samp_offset
        )

        job_tasks, job_cohints = calculate_tasks(
            job,
            self.cfg_params.n_ipp,
            self.cfg_params.num_cohints_per_file,
            self.exp_params.ipp_samps,
            self.sample_bounds,
        )

        progress_desc = "Coherent integrations"
        total = job_cohints
        extend_str_len = len(str(len(job_tasks)))
        total_num = str(len(job_tasks)).ljust(extend_str_len, " ")
        if self.progress:
            curr_num = "1".ljust(extend_str_len, " ")
            subprog_str = f"[file {curr_num}/{total_num}]"
            self.progress_bar = tqdm(
                desc=f"{progress_desc} {subprog_str}",
                total=total,
            )

        self._logger.info(f"starting job {job.idx}/{job.N} with {len(job_tasks)} tasks")

        results: AnalysedResult = {"dir": self.output_dir, "files": [], "data": {}}
        for idx, task_idx in enumerate(job_tasks):
            if self.progress_bar is not None:
                curr_num = f"{idx + 1}".ljust(extend_str_len, " ")
                subprog_str = f"[file {curr_num}/{total_num}]"
                self.progress_bar.set_description(
                    f"{progress_desc} {subprog_str}",
                )

            file_idx_sample = (
                task_idx
                * self.exp_params.ipp_samps
                * self.cfg_params.n_ipp
                * self.cfg_params.num_cohints_per_file
                + self.sample_bounds.start
            )

            task_data = self.process_task(
                task_idx=task_idx, file_idx_sample=file_idx_sample, bounds=self.sample_bounds
            )

            # Create directory if output is defined
            if self.output_dir is not None:
                output_path = Path(self.output_dir) / utils.get_filepath(epoch.start, file_idx_sample)
                # create directory
                dirname = Path(output_path).parent
                dirname.mkdir(parents=True, exist_ok=True)
            else:
                output_path = None

            self.save_task_data(
                file_idx_sample=file_idx_sample,
                out_data=task_data,
                results=results,
                filepath=output_path,
                clobber=clobber,
            )

        if self.progress_bar is not None:
            self.progress_bar.close()
        self._logger.info(f"finishing job {job.idx}/{job.N} with {len(job_tasks)} tasks")
        return results
