import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pyant.models.array import Array, ArrayParams

from hardtarget.constants import (
    AnalysisMethod,
    Impl,
    MethodLib,
)
from hardtarget.data_handling import Measurement
from hardtarget.process import get_analysis_process
from hardtarget.types import AnalysedResult, ArrayKwargs, GenericCfg, Job

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    from typing_extensions import Unpack
else:
    from typing import Unpack

def analyse(
    path: str | Path,
    config: str | Path | GenericCfg,
    method: AnalysisMethod,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    start_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    relative_time: bool = False,
    job: Optional[Job] = None,
    progress: bool = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
    **kwargs: Unpack[ArrayKwargs],
) -> AnalysedResult:
    """
    Perform matched filter analysis.

    Args:
        path: path to measurement file
        config: path to user config (.ini) or a config object matching the process e.g GMFCfgParams/DPTCfgParams
                assert that it is compatible with the process you want to run.
        method: Method to be used during the analysis
        method_lib (optional): Specific library to be used for the analysis method.
        implementation (optional): Implementation of the method to be used during the analysis (Numpy/Cuda/C),
                                   will override any implementation defined in the config params.
        rx_channel (optional): Specific rx channel to analyse, if none chosen all will be used from the meta data
        start_time (optional): Start time of analysis
        end_time (optional): End time of analysis
        relative_time (optional): If to use relative time
        job (optional): For parallelization the job to run the specific analysis on can be choosen.
        progress (optional): If a progress bar should be visualized.
        clobber (optional): If previous analysis should be overwritten.
        output (optional): Output directory for the analysed files, if None no files will be saved.
        **kwargs (optional): Extra data such as Beam and Beam parameters (needed for interferometry)

    """

    if job is None:
        job = Job(idx=0, N=1)

    # create measurement object to access measurement data
    measurement = Measurement(
        path=path,
        config=config,
        method=method,
        method_lib=method_lib,
        impl=implementation,
        rx_channel=rx_channel,
    )

    # determine process to run
    process_lib = get_analysis_process(method, method_lib)

    # Initiate process
    process = process_lib(
        config,
        exp_params=measurement.exp_params,
        cfg_params=measurement.cfg_params,
        pro_params=measurement.pro_params,
        epoch_bounds=measurement.epoch,
        func_get_data=measurement.extract_signals,
        func_get_pointing=measurement.pointing,
        output_dir=output,
        progress=progress,
        **kwargs,
    )

    # run analysis
    results = process.run(
        job=job,
        epoch=measurement.epoch,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        clobber=clobber,
    )

    return results


def target_estimation(
    path: str | Path,
    config: str | Path | GenericCfg,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    start_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    relative_time: bool = False,
    job: Optional[Job] = None,
    progress: bool = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
) -> AnalysedResult:
    """Wrapper around analyse for Target Estimations"""

    return analyse(
        path=path,
        config=config,
        method=AnalysisMethod.target_estimation,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        job=job,
        progress=progress,
        clobber=clobber,
        output=output,
    )


def optimize(
    path: str | Path,
    config: str | Path | GenericCfg,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    start_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    relative_time: bool = False,
    job: Optional[Job] = None,
    progress: bool = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
) -> AnalysedResult:
    """Wrapper around analyse for Target Estimation optimization"""

    return analyse(
        path=path,
        config=config,
        method=AnalysisMethod.optimize,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        job=job,
        progress=progress,
        clobber=clobber,
        output=output,
    )


def event_detection(
    path: str | Path,
    config: str | Path | GenericCfg,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    start_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    relative_time: bool = False,
    job: Optional[Job] = None,
    progress: bool = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
) -> AnalysedResult:
    """Wrapper around analyse for Event Detection"""

    return analyse(
        path=path,
        config=config,
        method=AnalysisMethod.event_detection,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        job=job,
        progress=progress,
        clobber=clobber,
        output=output,
    )


def direction_of_arrival(
    path: str | Path,
    config: str | Path | GenericCfg,
    array_beam: Array,
    beam_params: ArrayParams,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
    rx_channel: Optional[str | int] = None,
    start_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    relative_time: bool = False,
    job: Optional[Job] = None,
    progress: bool = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
) -> AnalysedResult:
    """Wrapper around analyse for Direction of Arrival"""

    return analyse(
        path=path,
        config=config,
        method=AnalysisMethod.direction_of_arrival,
        method_lib=method_lib,
        implementation=implementation,
        rx_channel=rx_channel,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        job=job,
        progress=progress,
        clobber=clobber,
        output=output,
        beam=array_beam,
        parameters=beam_params,
    )
