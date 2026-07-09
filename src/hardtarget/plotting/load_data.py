"""Collection of tools to load analysed data"""

import datetime as dt
import logging
import warnings
from collections.abc import Generator
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional, TypeVar, get_args, get_origin

import h5py
import numpy as np

from hardtarget.constants import AnalysisMethod, MethodAbbreviation
from hardtarget.process import get_analysis_process
from hardtarget.types import (
    ExpDef,
    GenericCfg,
    GenericOut,
    GenericPro,
    IsDataclass,
    OutputBase,
    ProParams,
)
from hardtarget.utils.h5_tools import get_analysed_h5_files
from hardtarget.utils.time_conversion import ts_from_str

logger = logging.getLogger(__name__)


def load_analysed_data(
    data_dir: str | Path | list[str] | list[Path],
    start_time: Optional[int | float | np.datetime64 | str] = None,
    end_time: Optional[int | float | np.datetime64 | str] = None,
    relative_time: bool = False,
    method: Optional[AnalysisMethod] = None,
    chunk_size: Optional[int] = None,
) -> Generator[tuple[GenericOut, ExpDef, GenericCfg, GenericPro], None, None]:
    """
    Loads and concatenates all analysed output data from 'data_dir'. Optionally specific timespans can be
    extracted, the result will be yielded in sizes of 'chunk_size' if given. Not that when merging a whole directory
    the data amount can cause quite the RAM usage.

    Args:
        data_dir: Directory containing the analysed output
        start_time (optional): start time, files containing data before this will be ignored. If relative time it should be declared in seconds.
        end_time (optional): end time, files containing data after this will be ignored. If relative time it should be declared in seconds.
        relative_time (optional): If relative time should be used.
        method (optional): Specific method to load data from, if not specified it will try to read all the available data.
        chunk_size (optional): If selected will split the path list in sizes of chunk_size. Each subgroup will
                               be yielded.
    Yields:
        Tuple of experiment params, process specific configuration, process specific params and analysed data.
    """

    if isinstance(start_time, str):
        try:
            start_time = ts_from_str(start_time)
        except ValueError:
            start_time = float(start_time)

    if isinstance(end_time, str):
        try:
            end_time = ts_from_str(end_time)
        except ValueError:
            end_time = float(end_time)

    paths = collect_paths(
        data_dir, start_time=start_time, end_time=end_time, relative_time=relative_time, method=method
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


def get_start_time(file: str | Path) -> float:
    file = Path(file)
    with h5py.File(file, "r") as hf:
        try:
            epoch_us = hf["OutArgs"]["epoch_us"][()]
            t = hf["OutArgs"]["t"][0]
            time_sec = epoch_us * 1e-6 + t
        except KeyError:
            time_sec = 0

    return time_sec


def collect_paths(
    folder: str | Path | list[str] | list[Path],
    start_time: Optional[int | float | np.datetime64] = None,
    end_time: Optional[int | float | np.datetime64] = None,
    relative_time: bool = False,
    method: Optional[AnalysisMethod] = None,
) -> list[Path]:
    """
    Sorts file according to start time, if requested filters out files that is not within the expected time.

    Args:
        folder: Directory containing the analyse output.
        start_time (optional): Start time, filter out any file before this time. If relative, declare in seconds.
        end_time (optional): End time, filter out any file after this time. If relative, declare in seconds.

    Returns:
        Time sorted list of output paths.
    """

    if isinstance(folder, list):
        folders = list(set(folder))

        fl = []
        for dir in folders:
            fl.extend(get_analysed_h5_files(dir, MethodAbbreviation[method] if method else None))  # type: ignore[arg-type]
    else:
        fl = get_analysed_h5_files(folder, MethodAbbreviation[method] if method else None)

    if not fl:
        return []

    fl = list(set(fl))
    fl.sort()

    if start_time or end_time:
        fl_epochs = [get_start_time(file) for file in fl]

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
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    dt64_t0 = np.datetime64(dt.datetime.fromtimestamp(epoch_unix, dt.timezone.utc))
            elif isinstance(start_time, np.datetime64):
                dt64_t0 = start_time  # type: ignore[assignment]
            elif isinstance(start_time, str):
                dt64_t0 = np.datetime64(int(ts_from_str(start_time) * 1e6), "us")
            else:
                dt64_t0 = np.datetime64(int(start_time * 1e6), "us")

            unix_t0 = dt64_t0.astype("datetime64[us]").astype("float64") * 1e-6

        if relative_time:
            if end_time is None:
                end = max_unix - epoch_unix
            else:
                end = float(end_time)
            unix_t1 = epoch_unix + end
        else:
            if end_time is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    dt64_t1 = np.datetime64(dt.datetime.fromtimestamp(max_unix, dt.timezone.utc))
            elif isinstance(end_time, np.datetime64):
                dt64_t1 = end_time  # type: ignore[assignment]
            elif isinstance(end_time, str):
                dt64_t1 = np.datetime64(int(ts_from_str(end_time) * 1e6), "us")
            else:
                dt64_t1 = np.datetime64(int(end_time * 1e6), "us")

            unix_t1 = dt64_t1.astype("datetime64[us]").astype("float64") * 1e-6

        fl = [file for file, ep in zip(fl, fl_epochs) if ep >= unix_t0 and ep <= unix_t1]

    return fl


GenericDataclass = TypeVar("GenericDataclass", bound=IsDataclass)


def extract_dataclass(
    file: h5py.File, dc_type: type[GenericDataclass], group_name: Optional[str] = None
) -> GenericDataclass:
    # If init is false for the dataclass, ignore it
    excluded_keys = [field.name for field in fields(dc_type) if not field.init]
    # Extract keys
    group_name = group_name if group_name else dc_type.__name__
    group = file[group_name]
    key_type = {f.name: f.type for f in fields(dc_type)}

    return dc_type(
        **{key: read_key(group, key, key_type[key]) for key in group.keys() if key not in excluded_keys}
    )


def collect_analysis_data(paths: list[Path]) -> tuple[GenericOut, ExpDef, GenericCfg, GenericPro]:
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
    if not paths:
        raise FileNotFoundError("No data present at given location")

    cfg_type, pro_type, out_type = get_process_types_from_file(paths[0])

    out_args = None
    exp_def: ExpDef | None = None
    cfg_params: GenericCfg | None = None
    pro_params: GenericPro | None = None

    out_buffer: list[OutputBase] = []
    for path in paths:
        with h5py.File(path, "r") as hf:
            if exp_def is None:
                exp_def = extract_dataclass(hf, ExpDef)
            if cfg_params is None:
                group = hf[cfg_type.__name__]
                cfg_params = cfg_type(**{key: read_key(group, key) for key in group.keys()})
            if pro_params is None:
                group = hf[pro_type.__name__]
                pro_params = pro_type(**{key: read_key(group, key) for key in group.keys()})

            if not out_args:
                out_args = extract_dataclass(hf["OutArgs"], out_type, "OutArgs")
            else:
                out_buffer.append(extract_dataclass(hf["OutArgs"], out_type, "OutArgs"))

    if not exp_def or not cfg_params or not pro_params or not out_args:
        raise FileNotFoundError(
            f"Exp present: {exp_def is not None}, Cfg present: {cfg_params is not None}, Pro present: {pro_params is not None}, Output present: {out_args is not None} "
        )

    # Concatenate output from all files to one
    out_args = out_args.copy_and_concatenate(out_buffer)

    return (
        out_args,
        exp_def,
        cfg_params,
        pro_params,
    )


def read_key(
    group: h5py.Group, key: str, d_type: Optional[Any] = None, logger: Optional[logging.Logger] = None
) -> Any:
    """h5py saves dataset string as byte strings, needs to be decoded"""
    data = group[key][()]
    if isinstance(data, bytes):
        data = data.decode()
    elif isinstance(data, np.ndarray) and (data.size == 0):
        if logger:
            logger.debug(f"{key} data is empty when loaded from file")
    elif isinstance(data, np.ndarray) and isinstance(data[0], bytes):
        data = [d.decode() for d in data]
    elif isinstance(data, np.integer):
        data = int(data)

    if any(get_origin(arg) is list for arg in get_args(d_type)) and type(data) is np.ndarray:
        # Integer lists are stored as numpy arrays, must be converted back
        data = data.tolist()

    return data


def get_process_types_from_file(path: Path) -> tuple[Any, Any, Any]:
    """
    From a given file determine what process was used and return the
    compatible data types

    Args:
        path: Path to analysed file
    Returns:
        tuple containing Configuration, Process and Output data type

    """

    with h5py.File(path, "r") as file:
        method = AnalysisMethod(file[f"{ProParams.method=}".split("=")[0].split(".")[1]].asstr()[()])
        method_lib = file[f"{ProParams.method_lib=}".split("=")[0].split(".")[1]].asstr()[()]
        # Retrive process matching the method and get the specific generic types for that process
        process = get_analysis_process(method, method_lib)

        return process.get_types()


def stack_analysed_data(
    data: dict[int, tuple[GenericOut, ExpDef, GenericCfg, GenericPro]],
) -> tuple[GenericOut, ExpDef, GenericCfg, GenericPro]:
    """
    Gather data from several different outputs to one common.

    Args:
        data: data output from analysed results. Shall be a dict containing a key for the start time with a tuple with all needed data.
    Returns:
        Tuple containting the output of all items in the inputs.
    """

    if not data:
        raise ValueError(
            "No data available in memory from analysis, could it have been stored in storage instead?"
        )

    sorted_data_list = list(dict(sorted(data.items())).values())
    _, exp_def, cfg, pro = sorted_data_list[0]

    out = None
    for output, _, _, _ in sorted_data_list:
        if not out:
            out = output
        else:
            out = out.copy_and_concatenate(output)

    if not exp_def or not cfg or not pro or not out:
        raise FileNotFoundError(
            f"Exp present: {exp_def is not None}, Cfg present: {cfg is not None}, Pro present: {pro is not None}, Output present: {out is not None} "
        )

    return out, exp_def, cfg, pro


"""
TODO: FIX THIS


def resume_analysis(outfile: Path, override: bool) -> bool:

    file_exists_do_not_override = outfile.is_file() and not override
    grid_calc_existing = method_type == MethodType.grid
    optimize_calc_existing = (method_type == MethodType.optimize) and (check_for_optimize_result(outfile))

    calc_existing = grid_calc_existing or optimize_calc_existing

    return file_exists_do_not_override and calc_existing

"""
