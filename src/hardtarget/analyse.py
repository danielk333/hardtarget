import datetime as dt
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pyant.models.array import Array, ArrayParams
from radardef import RadarDef
from radardef.components import DataLoader
from radardef.tools import mpi_tools
from radardef.types.types import ExpDef

from hardtarget.constants import (
    AnalysisMethod,
    Impl,
    MethodLib,
)
from hardtarget.process import get_analysis_process
from hardtarget.types import AnalysedResult, ArrayKwargs, GenericCfg

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    from typing_extensions import Unpack
else:
    from typing import Unpack


def analyse(
    data: str | Path | DataLoader,
    config: str | Path | GenericCfg | dict,
    method: AnalysisMethod,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    excluded_channels: Optional[list[str] | list[int]] = None,
    start_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    relative_time: bool = False,
    exp_def: Optional[ExpDef] = None,
    progress: bool | mpi_tools.CommBar = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
    sub_directory: Optional[str] = None,
    comm: mpi_tools.CommObject = mpi_tools.CommMock(),
    **kwargs: Unpack[ArrayKwargs],
) -> AnalysedResult:
    """
    Perform matched filter analysis.
    Hardtarget analysis supports multiprocessing,

    Args:
        data: path to measurement file or data loader
        config: path to user config (.ini) or a config object matching the process e.g GMFCfgParams/DPTCfgParams
                assert that it is compatible with the process you want to run.
        method: Method to be used during the analysis
        method_lib (optional): Specific library to be used for the analysis method.
        implementation (optional): Implementation of the method to be used during the analysis (Numpy/Cuda/C),
                                   will override any implementation defined in the config params.
        rx_channel (optional): Specific rx channel to analyse, if none chosen all will be used from the meta data
        start_time (optional): Start time of analysis, datetime object, date string or seconds since epoch.
        end_time (optional): End time of analysis,datetime object, date string or seconds since epoch.
        relative_time (optional): If to use relative time, start and end time must be specified in int or float then.
        exp_def (optional): If working with custom experiments it is needed to be able to load the data.
        progress (optional): If a progress bar should be visualized.
        clobber (optional): If previous analysis should be overwritten.
        output (optional): Output directory for the analysed files, if None no files will be saved.
                           Path will be output/YYYY-MM-DD/HH-00-00
        sub_directory (optional): Sub directory, stated if there is a need to store data in a specific subfolder in the output path.
                                 More specifically output/YYYY-MM-DD/HH-00-00/sub_directory/
        comm (optional): Mpi communication object if analysis is expected to run on MPI or a specific Comm object.
        **kwargs (optional): Extra data such as Beam and Beam parameters (needed for interferometry)

    """

    if comm.rank == 0:
        # Access data
        if isinstance(data, str) or isinstance(data, Path):
            data_loader = RadarDef().load_data(Path(data), exp_def=exp_def)
            if data_loader is None:
                raise ValueError(f"Not possible to load data file from: {data}")
        else:
            data_loader = data

        # Determine process to run
        process_lib = get_analysis_process(method, method_lib)

        # Initiate process
        process = process_lib(
            config=config,
            data=data_loader,
            method_lib=method_lib,
            impl=implementation,
            rx_channel=rx_channel,
            excluded_channels=excluded_channels,
            output_dir=output,
            **kwargs,
        )
    else:
        process = None

    process = comm.bcast(process, root=0)

    # run analysis
    results = process.run(
        comm_rank=comm.rank,
        comm_size=comm.size,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        sub_directory=sub_directory,
        clobber=clobber,
        progress=progress,
    )

    if comm.size > 1:
        all_results = comm.gather(results, root=0)

        if comm.rank == 0:
            results = merge_results_from_all_ranks(all_results)  # type: ignore[arg-type]
        else:
            results = None

        results = comm.bcast(results, root=0)
        comm.barrier()

    return results


def target_estimation(
    data: str | Path | DataLoader,
    config: str | Path | GenericCfg | dict,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    excluded_channels: Optional[list[str] | list[int]] = None,
    start_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    relative_time: bool = False,
    exp_def: Optional[ExpDef] = None,
    progress: bool | mpi_tools.CommBar = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
    sub_directory: Optional[str] = None,
    comm: mpi_tools.CommObject = mpi_tools.CommMock(),
) -> AnalysedResult:
    """Wrapper around analyse for Target Estimations"""

    return analyse(
        data=data,
        config=config,
        method=AnalysisMethod.target_estimation,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        excluded_channels=excluded_channels,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        exp_def=exp_def,
        progress=progress,
        clobber=clobber,
        output=output,
        sub_directory=sub_directory,
        comm=comm,
    )


def optimize(
    data: str | Path | DataLoader,
    config: str | Path | GenericCfg | dict,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    excluded_channels: Optional[list[str] | list[int]] = None,
    start_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    relative_time: bool = False,
    exp_def: Optional[ExpDef] = None,
    progress: bool | mpi_tools.CommBar = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
    sub_directory: Optional[str] = None,
    comm: mpi_tools.CommObject = mpi_tools.CommMock(),
) -> AnalysedResult:
    """Wrapper around analyse for Target Estimation optimization"""

    return analyse(
        data=data,
        config=config,
        method=AnalysisMethod.optimize,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        excluded_channels=excluded_channels,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        exp_def=exp_def,
        progress=progress,
        clobber=clobber,
        output=output,
        sub_directory=sub_directory,
        comm=comm,
    )


def echo_search(
    data: str | Path | DataLoader,
    config: str | Path | GenericCfg | dict,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    excluded_channels: Optional[list[str] | list[int]] = None,
    start_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    relative_time: bool = False,
    exp_def: Optional[ExpDef] = None,
    progress: bool | mpi_tools.CommBar = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
    sub_directory: Optional[str] = None,
    comm: mpi_tools.CommObject = mpi_tools.CommMock(),
) -> AnalysedResult:
    """Wrapper around analyse for echo search"""

    return analyse(
        data=data,
        config=config,
        method=AnalysisMethod.echo_search,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        excluded_channels=excluded_channels,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        exp_def=exp_def,
        progress=progress,
        clobber=clobber,
        output=output,
        sub_directory=sub_directory,
        comm=comm,
    )


def direction_of_arrival(
    data: str | Path | DataLoader,
    config: str | Path | GenericCfg | dict,
    array_beam: Array,
    beam_params: ArrayParams,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    excluded_channels: Optional[list[str] | list[int]] = None,
    start_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | float | str | dt.datetime] = None,
    relative_time: bool = False,
    exp_def: Optional[ExpDef] = None,
    progress: bool | mpi_tools.CommBar = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
    sub_directory: Optional[str] = None,
    comm: mpi_tools.CommObject = mpi_tools.CommMock(),
) -> AnalysedResult:
    """Wrapper around analyse for Direction of Arrival"""

    return analyse(
        data=data,
        config=config,
        method=AnalysisMethod.direction_of_arrival,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        excluded_channels=excluded_channels,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        exp_def=exp_def,
        progress=progress,
        clobber=clobber,
        output=output,
        sub_directory=sub_directory,
        beam=array_beam,
        parameters=beam_params,
        comm=comm,
    )


def merge_results_from_all_ranks(results: list[AnalysedResult]) -> AnalysedResult:

    result: AnalysedResult = {"dir": results[0]["dir"], "file_dir": [], "files": [], "data": {}}

    for res in results:
        for key, value in res.items():
            if key == "file_dir":
                if isinstance(value, list):
                    if key not in result:
                        result["file_dir"] = value
                    else:
                        if value not in result["file_dir"]:
                            result["file_dir"].extend(value)
            if key == "files":
                if isinstance(value, list):
                    if key not in result:
                        result["files"] = value
                    else:
                        result["files"].extend(value)
            elif key == "data":
                if isinstance(value, dict):
                    if key not in result:
                        result["data"] = value
                    else:
                        result["data"].update(value)
            elif key == "dir":
                if isinstance(value, Path) or isinstance(value, str) or value is None:
                    if key not in result:
                        result["dir"] = value
                    else:
                        if value != result["dir"]:
                            raise Exception("Output data stored in different directories")

    return result
