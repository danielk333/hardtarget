import datetime as dt
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hardtarget.data_handling import Measurement
from hardtarget.matched_filter import get_analysis_process
from hardtarget.types.constants import EstimationMethod, Impl
from hardtarget.types.types import AnalysedResult, GenericCfg, Job


def analyse(
    path: str | Path,
    config: str | Path | GenericCfg,
    rx_channel: Optional[str | int] = None,
    start_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    end_time: Optional[np.datetime64 | int | str | dt.datetime] = None,
    relative_time: bool = False,
    job: Optional[Job] = None,
    progress: bool = False,
    clobber: bool = True,
    output: Optional[str | Path] = None,
    method: Optional[EstimationMethod] = None,
    implementation: Optional[Impl] = None,
    logger: Optional[logging.Logger] = None,
) -> AnalysedResult:
    """
    Perform matched filter analysis.

    Args:
        path: path to measurement file
        config: path to user config (.ini) or a config object matching the process e.g GMFCfgParams/DPTCfgParams
                assert that it is compatible with the process you want to run.
        rx_channel: Specific rx channel to analyse, if none chosen all will be used from the meta data
        start_time: Start time of analysis
        end_time: End time of analysis
        relative_time: If to use relative time
        job: For parallelization the job to run the specific analysis on can be choosen.
        progress: If a progress bar should be visualized.
        clobber: If previous analysis should be overwritten.
        output: Output directory for the analysed files, if None no files will be saved.
        method: Method to be used during the analysis, will override any estimation method defined in the
                config params.
        implementation: Implementation of the method to be used during the analysis (Numpy/Cuda/C),
                        will override any implementation defined in the config params.
        logger: Specific logger to log on for debugging.

    """

    if job is None:
        job = Job(idx=0, N=1)

    if logger is None:
        logger = logging.getLogger(__name__)

    # create measurement object to access measurement data
    measurement = Measurement(
        path=path,
        config=config,
        rx_channel=rx_channel,
        impl=implementation,
        est_method=method,
    )

    # determine process to run
    process_lib = get_analysis_process(measurement.analysis_method)

    # Initiate process
    process = process_lib(
        config,
        exp_params=measurement.exp_params,
        cfg_params=measurement.cfg_params,
        pro_params=measurement.pro_params,
        epoch_bounds=measurement.epoch,
        func_get_data=measurement.extract_signals,
        func_get_pointing=measurement.pointing,
        lib=measurement.estimation_method,
        output_dir=output,
        progress=progress,
    )

    # run analysis
    results = process.run(
        job=job,
        channel_bounds=measurement.rx_sample_bounds,
        epoch=measurement.epoch,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
        clobber=clobber,
    )

    return results
