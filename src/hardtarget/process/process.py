"""
Process, the class represents an arbitrary radar data analysis process. This is a skeleton for the process
and controls the general flow. To create a complete process the class needs to be inherited and the
abstract methods filled in. It supports a variety of datatypes.
"""

import datetime as dt
import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, Type

import numpy as np
from radardef.components import DataLoader
from radardef.types import Pointing
from tqdm import tqdm

import hardtarget.process.utils as utils
from hardtarget.constants import AnalysisMethod, Impl, MethodLib
from hardtarget.data_handling import dump_params_to_file
from hardtarget.data_simulation.tx_model import tx_signal_model
from hardtarget.process.configuration import (
    compute_process_params,
    extract_config_from_dict,
    load_config_params,
)
from hardtarget.process.utils import calculate_tasks, sample_interval_to_closest_ipp
from hardtarget.types import (
    AnalysedResult,
    ArrayKwargs,
    Bounds,
    CfgParams,
    DataItem,
    ExpDef,
    ExtractedSignals,
    GenericCfg,
    GenericLib,
    GenericOut,
    GenericPro,
    GenericVars,
    ProParams,
)
from hardtarget.utils.time_conversion import time_interval_to_sample_bound, ts_from_str

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    from typing_extensions import Unpack
else:
    from typing import Unpack

try:
    # Only available from python 3.12
    from types import get_original_bases  # type: ignore[attr-defined,unused-ignore]

    def orig_bases(cls: Type) -> tuple[Any, ...]:
        return get_original_bases(cls)

except ImportError:

    def orig_bases(cls: Type) -> tuple[Any, ...]:
        return cls.__orig_bases__


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
        config: path to user config file or CfgParams object matching the specfic process
        data: Data loader to access the measurement data
        method_lib (optional): Specific method library
        impl (optional): Implementation to be used during analyse (C/Cuda/Numpy)
        rx_channel (optional): If only a specific rx channel should be analysed, if None all rx channels will be used for analysis.
        excluded_channels (optional): Rx channels to ignore if the data contains multiple channels.
        progress (optional): Enable progress bar.
        output_dir (optional): Path to output directory, if none data will only be stored in ram
        **kwargs: Extra data such as Beam and Beam parameters (needed for interferometry)

    """

    method: AnalysisMethod

    def __init__(
        self,
        config: str | Path | GenericCfg,
        data: DataLoader,
        method_lib: Optional[MethodLib] = None,
        impl: Optional[Impl] = None,
        rx_channel: Optional[str | int] = None,
        excluded_channels: Optional[list[str] | list[int]] = None,
        output_dir: Optional[str | Path] = None,
        progress: bool = False,
        **kwargs: Unpack[ArrayKwargs],
    ) -> None:
        # Local logger
        self._logger = logging.getLogger(__name__)

        # Experiment definition and data acquisition
        self.data = data
        self.exp_params = self.data.experiment
        self._rx_channel, self._tx_channel = self.extract_channels(self.exp_params, rx_channel)
        self._excluded_channels = excluded_channels if excluded_channels is not None else []

        # Define configuration
        self.raw_config = config
        if isinstance(config, CfgParams):
            self.cfg_params = config
        elif isinstance(config, dict):
            cfg_type, _, _ = self.get_types()
            self.cfg_params = extract_config_from_dict(config, cfg_type)
        else:
            cfg_base = load_config_params(Path(config))
            self.cfg_params = self.get_conf_params(Path(config), cfg_base)

        # Define library to be used during process
        self.lib, lib_name, impl = self.get_analysis_lib(method_lib, impl)

        # Derive process parameters
        pro_base = compute_process_params(
            self.exp_params,
            self.cfg_params,
            analysis_method=self.method,
            method_lib=lib_name,
            implementation=impl,
        )
        self.pro_params = self.get_process_params(self.exp_params, self.cfg_params, pro_base)

        # Other needed parameters
        t_start_usec, t_end_usec = self.data.epoch_bounds
        self.epoch = Bounds(int(t_start_usec), int(t_end_usec))
        self.progress = progress
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self.progress_bar = None
        self.store_mode = "w"
        self.store_params = True

        self.kwargs = kwargs
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post init"""
        pass

    @classmethod
    def get_types(cls) -> tuple[type[GenericCfg], type[GenericPro], type[GenericOut]]:
        """
        Class method to get the class specific defined types

        Return:
            Configuration type, Process type and Output type
        """

        if cls.__bases__[0] != Process:
            # For target estimation processes there is a double inheritance case
            _, _, _, out_type, _ = orig_bases(cls.__bases__[0])[0].__args__
            cfg_type, pro_type = orig_bases(cls)[0].__args__
        else:
            # Get process base types
            generic_types = orig_bases(cls)[0].__args__
            cfg_type, pro_type, _, out_type, _ = generic_types

        return cfg_type, pro_type, out_type

    @abstractmethod
    def get_analysis_lib(
        self, lib: MethodLib | None, impl: Impl | None
    ) -> tuple[GenericLib, MethodLib, Impl]:
        """Get specific library to run analysis"""
        pass

    @abstractmethod
    def get_conf_params(self, cfg_path: Path, cfg_params: CfgParams) -> GenericCfg:
        """Abstract method, process specific configuration parameters"""
        pass

    @abstractmethod
    def get_process_params(
        self, exp_params: ExpDef, cfg_params: GenericCfg, pro_params: ProParams
    ) -> GenericPro:
        """Abstract method, process specific parameters"""
        pass

    @abstractmethod
    def analyse_ipps(self, start_sample: int) -> GenericVars:
        """
        Abstract method, shall analyse the interpulse periods from start sample.

        Args:
            start_sample: sample index to start analysis at.

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
        exp_params: ExpDef,
        cfg_params: GenericCfg,
        pro_params: GenericPro,
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
        return self.generate_output(
            all_vars, file_idx_sample, self.exp_params, self.cfg_params, self.pro_params
        )

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
        comm_rank: int,
        comm_size: int,
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
        if start_time or end_time:
            sample_bounds = time_interval_to_sample_bound(
                start_time=start_time,
                end_time=end_time,
                time_bounds=self.epoch,
                sample_rate=self.exp_params.sample_rate,
                relative_time=relative_time,
            )
        else:
            sample_bounds = Bounds(*self.data.bounds(self.exp_params.rx_channels[0]))

        # round off to closest ipp, starting in the middle of a ipp will cause issues to the analysis
        sample_bounds = sample_interval_to_closest_ipp(
            sample_bounds=sample_bounds, ipp_samps=self.exp_params.ipp_samps
        )

        # add potential sample offset
        sample_bounds = Bounds(
            sample_bounds.start + self.cfg_params.samp_offset, sample_bounds.end + self.cfg_params.samp_offset
        )

        job_tasks, job_cohints = calculate_tasks(
            comm_rank,
            comm_size,
            self.cfg_params.n_ipp,
            self.cfg_params.num_cohints_per_file,
            self.exp_params.ipp_samps,
            sample_bounds,
        )

        progress_desc = "Coherent integrations"
        total = job_cohints
        extend_str_len = len(str(len(job_tasks)))
        total_num = str(len(job_tasks)).ljust(extend_str_len, " ")
        if self.progress:
            curr_num = "1".ljust(extend_str_len, " ")
            subprog_str = f"[file {curr_num}/{total_num}]"
            self.progress_bar = tqdm(
                desc=f"{self.method}: {progress_desc} {subprog_str}",
                total=total,
            )

        self._logger.info(f"starting job {comm_rank}/{comm_size} with {len(job_tasks)} tasks")

        results: AnalysedResult = {"dir": self.output_dir, "files": [], "data": {}}
        for idx, task_idx in enumerate(job_tasks):
            if self.progress_bar is not None:
                curr_num = f"{idx + 1}".ljust(extend_str_len, " ")
                subprog_str = f"[file {curr_num}/{total_num}]"
                self.progress_bar.set_description(
                    f"{progress_desc} {subprog_str}",
                )

            # Calculate start sample of task
            file_idx_sample = (
                task_idx
                * self.exp_params.ipp_samps
                * self.cfg_params.n_ipp
                * self.cfg_params.num_cohints_per_file
                + sample_bounds.start
            )

            # Create directory and define filename.
            if self.output_dir is not None:
                output_path = Path(self.output_dir) / utils.get_filepath(self.epoch.start, file_idx_sample)
                # create directory
                dirname = Path(output_path).parent
                dirname.mkdir(parents=True, exist_ok=True)
            else:
                output_path = None

            # If file exists and clobber off, skip analysis.
            if output_path and output_path.is_file() and not clobber:
                results["files"].append(output_path.name)
                self._logger.debug(
                    f"File already existing and clobber is off, file: {output_path.name} is skipped."
                )
                if self.progress_bar:
                    self.progress_bar.update(self.cfg_params.num_cohints_per_file)
            # Else run analysis
            else:
                task_data = self.process_task(
                    task_idx=task_idx, file_idx_sample=file_idx_sample, bounds=sample_bounds
                )

                self.save_task_data(
                    file_idx_sample=file_idx_sample,
                    out_data=task_data,
                    results=results,
                    filepath=output_path,
                    clobber=clobber,
                )

        if self.progress_bar:
            self.progress_bar.close()
        self._logger.info(f"finishing job {comm_rank}/{comm_size} with {len(job_tasks)} tasks")
        return results

    def get_pointing(self, start_sample: int) -> Pointing:
        """Pointing data to define radar pointing direction in spherical coordinates"""
        return self.data.pointing(start_sample)

    def extract_channels(
        self, exp_params: ExpDef, rx_channel: Optional[int | str] = None
    ) -> tuple[int | str | None, int | str | None]:
        """
        From experiment parameters and requested rx channel extract the correct ones from the data file
        """

        _tx_channel = exp_params.tx_channel
        _rx_channel = None
        if rx_channel is not None:
            _rx_channel = rx_channel
            if _rx_channel not in exp_params.rx_channels:
                raise ValueError(f"rx_channel: {rx_channel} is not a valid channel in the measurement file")
        elif len(exp_params.rx_channels) == 1:
            # Only one available rx_channel during the experiment, thus we can declare it here
            _rx_channel = exp_params.rx_channels[0]

        return _rx_channel, _tx_channel

    def get_data(
        self,
        start_sample: int,
        read_length: int,
        sum_rx_channels: bool = True,
        sub_resolution: int = 1,
    ) -> ExtractedSignals:
        """
        Extract rx and tx data at the given sample.

        Args:
            start_sample: Start sample
            read_length: Amount of sample to read from the start sample
            sum_rx_channels (optional): If multiple rx channels available they will all be summed.
        Returns:
            Rx and Tx samples, either as a array (read_length,) or if multiple channels requested
            (n_channels, read_length).
            If no tx channel is available a tx model will be used to simulate the tx signal
        """

        # if no rx channel specified all channels will be summed for full analysis
        if sum_rx_channels:
            if self._rx_channel is None:
                ipp = np.zeros((read_length,), dtype=np.complex128)
                for chnl in self.data.channels:
                    if chnl != self._tx_channel and chnl not in self._excluded_channels:
                        ipp += self.data.read(chnl, start_sample, read_length)
            else:
                ipp = self.data.read(self._rx_channel, start_sample, read_length)
            # Extract rx samples
            rx = ipp[self.pro_params.rx_stencil].copy()
        else:
            ipp = np.zeros((len(self.exp_params.rx_channels), read_length), dtype=np.complex128)
            for i, chnl in enumerate[int | str](self.data.channels):
                ipp[i, :] = self.data.read(chnl, start_sample, read_length)
            # Extract rx samples
            rx = ipp[:, self.pro_params.rx_stencil].copy()

        # Extracting tx data
        if self._tx_channel != self._rx_channel and self._tx_channel is not None:
            tx = self.data.read(self._tx_channel, start_sample, read_length)
            tx = np.broadcast_to(tx.reshape((tx.size, 1)), (tx.size, sub_resolution))
        elif self._tx_channel == self._rx_channel and self._tx_channel is not None:
            tx = ipp.copy()
            tx = np.broadcast_to(tx.reshape((tx.size, 1)), (tx.size, sub_resolution))
        else:
            assert self.exp_params.code is not None, (
                "No code available from the metadata, not possible to simulate tx"
            )
            tx = tx_signal_model(
                code=self.exp_params.code,
                baud_length_usec=self.exp_params.baud_length_usec,
                t_samp_usec=self.exp_params.t_samp_usec,
                tx_start_samp=int(self.exp_params.t_tx_start_usec / self.exp_params.t_samp_usec),
                start_samp=(start_sample % self.exp_params.ipp_samps) - self.cfg_params.samp_offset,
                read_length=read_length,
                ipp_samps=self.exp_params.ipp_samps,
                sub_resolution=sub_resolution,
                kind="linear",
            )
        tx = tx[self.pro_params.tx_stencil, :]

        return ExtractedSignals(
            tx=tx.astype(np.complex64), rx=rx.astype(np.complex64), ipp=ipp.astype(np.complex64)
        )
