"""Collection of tools to load analysed data"""

import datetime as dt
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Type

import h5py
import numpy as np
import numpy.typing as npt

from hardtarget.matched_filter import get_analysis_process
from hardtarget.matched_filter.types import MFOptimizeOutArgs, MFOutArgs
from hardtarget.types.constants import AnalysisMethod
from hardtarget.types.types import ExpParams, GenericCfg, GenericOut, GenericPro, ProParams
from hardtarget.utils.h5_tools import get_analysed_h5_files

try:
    # Only available from python 3.12
    from types import get_original_bases  # type: ignore[attr-defined,unused-ignore]

    def orig_bases(cls: Type) -> tuple[Any, ...]:
        return get_original_bases(cls)

except ImportError:

    def orig_bases(cls: Type) -> tuple[Any, ...]:
        return cls.__orig_bases__


logger = logging.getLogger(__name__)


def load_optimized_data(path: str | Path) -> MFOptimizeOutArgs | None:

    paths = get_analysed_h5_files(path)
    paths.sort()

    main_data: dict[str, npt.NDArray[np.float64]] = {}
    for path in paths:
        with h5py.File(path, "r") as hf:
            group = hf["OutArgs"]  # TODO: Update to proper type
            tmp_data = {}

            for key in MFOptimizeOutArgs._fields:
                if key in group:
                    tmp_data[key] = group[key][()]
                    if key in main_data:
                        main_data[key] = np.append(main_data[key], tmp_data[key], axis=0)
                    else:
                        main_data[key] = tmp_data[key]
    if not main_data:
        return None
    else:
        return MFOptimizeOutArgs(**main_data)


def load_analysed_data(
    data_dir: str | Path,
    start_time: Optional[int | np.datetime64] = None,
    end_time: Optional[int | np.datetime64] = None,
    relative_time: bool = False,
    chunk_size: Optional[int] = None,
) -> Generator[tuple[GenericOut, ExpParams, GenericCfg, GenericPro], None, None]:
    """
    Loads and concatenates all analysed output data from 'data_dir'. Optionally specific timespans can be
    extracted, the result will be yielded in sizes of 'chunk_size' if given.

    Args:
        folder: Directory containing the analysed output
        start_time (optional): start time, files containing data before this will be ignored.
        end_time (optional): end time, files containing data after this will be ignored.
        relative_time (optional): If relative time should be used.
        chunk_size (optional): If selected will split the path list in sizes of chunk_size. Each subgroup will
                               be yielded.
    Yields:
        Tuple of experiment params, process specific configuration, process specific params and analysed data.
    """

    paths = collect_paths(
        data_dir,
        start_time=start_time,
        end_time=end_time,
        relative_time=relative_time,
    )

    paths.sort()
    pth_num = len(paths)
    if chunk_size is None:
        chunks = 1
        chunk_size = pth_num
    else:
        chunks = pth_num // chunk_size + 1
    for ind in range(chunks):
        sub_paths = paths[(ind * chunk_size) : ((ind + 1) * chunk_size)]
        yield collect_analysis_data(sub_paths)


def collect_paths(
    folder: str | Path,
    start_time: Optional[int | np.datetime64] = None,
    end_time: Optional[int | np.datetime64] = None,
    relative_time: bool = False,
) -> list[Path]:
    """
    Sorts file according to start time, if requested filters out files that is not within the expected time.

    Args:
        folder: Directory containing the analyse output.
        start_time (optional): Start time, filter out any file before this time.
        end_time (optional): End time, filter out any file after this time.

    Returns:
        Time sorted list of output paths.
    """

    fl = get_analysed_h5_files(folder)
    fl.sort()
    fl_epochs = [int(file.stem.split("-")[1]) * 1e-6 for file in fl]

    epoch_unix = fl_epochs[0]
    max_unix = fl_epochs[-1]

    if relative_time:
        if start_time is None:
            start = 0.0
        else:
            start = float(start_time)
        unix_t0 = epoch_unix + start
    else:
        if start_time is None:
            dt64_t0 = np.datetime64(dt.datetime.fromtimestamp(epoch_unix, dt.timezone.utc))
        elif isinstance(start_time, np.datetime64):
            dt64_t0 = start_time  # type: ignore[assignment]
        else:
            dt64_t0 = np.datetime64(start_time, "us")

        unix_t0 = dt64_t0.astype("datetime64[us]").astype("int64") * 1e-6

    if relative_time:
        if end_time is None:
            end = max_unix - epoch_unix
        else:
            end = float(end_time)
        unix_t1 = epoch_unix + end
    else:
        if end_time is None:
            dt64_t1 = np.datetime64(dt.datetime.fromtimestamp(max_unix, dt.timezone.utc))
        elif isinstance(end_time, np.datetime64):
            dt64_t1 = end_time  # type: ignore[assignment]
        else:
            dt64_t1 = np.datetime64(end_time, "us")

        unix_t1 = dt64_t1.astype("datetime64[us]").astype("int64") * 1e-6

    fl = [file for file, ep in zip(fl, fl_epochs) if ep >= unix_t0 and ep <= unix_t1]

    return fl


def collect_analysis_data(paths: list[Path]) -> tuple[GenericOut, ExpParams, GenericCfg, GenericPro]:
    """
    From the stored analysed data, determines what method was used during analysis and loads
    the data to the appropriate analys process specfic types. If mulitple files are given, the data is merged
    to one object

    Args:
        paths: List of paths to load data from and merge.

    Returns:
        Process specific types with the experiment data, configuration data, process data and the analysed
        output.
    """

    with h5py.File(paths[0], "r") as file:
        # Get analysis method from the process parameters
        method = AnalysisMethod(file[f"{ProParams.method=}".split("=")[0].split(".")[1]].asstr()[()])
        # Retrive process matching the method and get the specific generic types for that process
        generic_types = orig_bases(get_analysis_process(method))[0].__args__  # type: ignore[arg-type]
        cfg_type, pro_type, _, out_type, _ = generic_types

    out_args: dict[str, Any] = {}
    exp_params: ExpParams | None = None
    cfg_params: GenericCfg | None = None
    pro_params: GenericPro | None = None

    for path in paths:
        out_tmp = {}
        with h5py.File(path, "r") as hf:
            group = hf["OutArgs"]  # TODO: Update to proper type
            out_tmp = {key: group[key][()] for key in out_type._fields}

            def read_key(group: h5py.Group, key: str) -> Any:
                """h5py saves dataset string as byte strings, needs to be decoded"""
                data = group[key][()]
                if isinstance(data, bytes):
                    data = data.decode()
                elif isinstance(data, np.ndarray) and isinstance(data[0], bytes):
                    data = [d.decode() for d in data]
                return data

            if exp_params is None:
                group = hf[ExpParams.__name__]
                exp_params = ExpParams(**{key: read_key(group, key) for key in group.keys()})
            if cfg_params is None:
                group = hf[cfg_type.__name__]
                cfg_params = cfg_type(**{key: read_key(group, key) for key in group.keys()})
            if pro_params is None:
                group = hf[pro_type.__name__]
                pro_params = pro_type(**{key: read_key(group, key) for key in group.keys()})

        def _append_data(main_data: dict, tmp_data: dict, logger: logging.Logger) -> dict:
            if not main_data:
                for key in tmp_data:
                    logger.debug(f"Init mat {key}: {tmp_data[key].shape} [{tmp_data[key].dtype}]")
                    main_data[key] = tmp_data[key]
            else:
                for key in tmp_data:
                    # only interested in the epoch start of the measurement TODO: adjust this
                    if key == f"{MFOutArgs.epoch=}".split("=")[0].split(".")[1]:
                        continue
                    if isinstance(tmp_data[key], np.ndarray):
                        logger.debug(f"Append mat {key}: {tmp_data[key].shape} [{tmp_data[key].dtype}]")
                        main_data[key] = np.append(main_data[key], tmp_data[key], axis=0)
                    else:
                        logger.debug(f"Add {key}: {tmp_data[key].dtype}")
                        main_data[key] = main_data[key] + tmp_data[key]
            return main_data

        out_args = _append_data(out_args, out_tmp, logger)

    return (
        out_type(**out_args),
        exp_params if exp_params is not None else ExpParams(),
        cfg_params if cfg_params is not None else cfg_type(),
        pro_params if pro_params is not None else pro_type(),
    )


"""
TODO: FIX THIS


def resume_analysis(outfile: Path, override: bool) -> bool:

    file_exists_do_not_override = outfile.is_file() and not override
    grid_calc_existing = method_type == MethodType.grid
    optimize_calc_existing = (method_type == MethodType.optimize) and (check_for_optimize_result(outfile))

    calc_existing = grid_calc_existing or optimize_calc_existing

    return file_exists_do_not_override and calc_existing

"""
