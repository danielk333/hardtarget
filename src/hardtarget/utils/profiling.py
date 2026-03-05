import datetime
import logging
import os
import pathlib
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Pattern

from hardtarget.utils import global_mpi

try:
    import yappi

    YAPPI_AVAILABLE = True
except ImportError:
    YAPPI_AVAILABLE = False

# if code is run by mpi, mpi is already imported)
comm = global_mpi.get_mpi()


PACKAGE_NAME = "hardtarget"
PACKAGE_PATH = str(pathlib.Path(__file__).parent)
START_TIME: float = 0.0

logger = logging.getLogger(__name__)


def get_logging_level(verbosity: int) -> int:
    if verbosity <= 0:
        return logging.WARNING
    elif verbosity == 1:
        return logging.INFO
    elif verbosity >= 2:
        return logging.DEBUG
    else:
        raise ValueError(f"Verbosity '{verbosity}' not recognized")


def check_parallel() -> tuple[bool, int | None]:
    if comm is None:
        return False, None
    else:
        if comm.size > 1:
            return True, comm.rank
        else:
            return False, None


def modify_subloggers(matching_regex: str | Pattern[str]) -> None:
    raise NotImplementedError()
    _ = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if re.search(matching_regex, name) is not None
    ]


def setup_loggers(
    log_folder: Optional[Path | str] = None, stdout: bool = True, verbosity: int = 0
) -> logging.Logger:
    parallel, rank = check_parallel()
    if parallel:
        formatter = logging.Formatter(f"RANK{rank} - %(asctime)s %(levelname)s %(name)s - %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
    formatter.default_msec_format = "%s.%03d"

    handlers: list[logging.StreamHandler | logging.FileHandler] = []
    if stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_folder is not None:
        if not isinstance(log_folder, pathlib.Path):
            log_folder = pathlib.Path(log_folder)
        now = datetime.datetime.now()
        datetime_str = now.strftime("%Y-%m-%d_at_%H-%M")
        if not log_folder.is_dir():
            assert not log_folder.is_file(), f"Cannot use '{log_folder}', is a file not a folder"
            log_folder.mkdir(parents=True)

            if parallel:
                log_fname = str(log_folder / f"hardtarget_{datetime_str}_RANK{rank}.log")
            else:
                log_fname = str(log_folder / f"hardtarget_{datetime_str}.log")

        handlers.append(logging.FileHandler(log_fname))

    liblogger = logging.getLogger("hardtarget")
    # So that the logging does not propagate up to parent loggers
    # but settings do propagate down to child loggers
    liblogger.propagate = False

    level = get_logging_level(verbosity)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        liblogger.setLevel(level)
        liblogger.addHandler(handler)

    return liblogger


def _path_to_module(path: str) -> str:
    if PACKAGE_PATH not in path:
        return ""
    stem = pathlib.Path(path).stem
    path = PACKAGE_NAME + os.sep + path.replace(PACKAGE_PATH, "").strip(os.sep)
    module = path.split(os.sep)
    module[-1] = stem
    return ".".join(module)


def check_yappi(func: Callable) -> Callable:
    def checked_func(*args: Any, **kwargs: Any) -> Callable:
        if YAPPI_AVAILABLE is False:
            raise ImportError("'yappi' not installed, please install to profile")
        return func(*args, **kwargs)

    return checked_func


@check_yappi
def profile() -> None:
    global START_TIME
    START_TIME = time.time()
    yappi.set_clock_type("cpu")
    yappi.start()


@check_yappi
def get_profile(modules: Optional[list[str]] = None) -> tuple[yappi.YFuncStats, float]:
    if modules is None:
        modules = [PACKAGE_NAME]
    stats = yappi.get_func_stats(
        filter_callback=lambda x: any(list(_path_to_module(x.module).startswith(mod) for mod in modules)),
    )
    stats = stats.sort("ttot", "desc")

    total = time.time() - START_TIME
    return stats, total


def print_profile(
    stats: yappi.YFuncStats, total: Optional[float] = None, max_rows: Optional[int] = None
) -> None:
    header = [
        "Name",
        "Module",
        "Calls",
        "Total [s]",
        "Function [s]",
        "Average [s]",
    ]
    column_sizes = [len(title) for title in header]
    formats = [""] * 3 + ["1.4e"] * 3

    if total is not None:
        header += ["Total [%]"]
        column_sizes += [len(header[-1])]
        formats += ["2.3f"]

    if total is None:
        total = 1

    for ind in range(3, len(header)):
        if column_sizes[ind] < 6:
            column_sizes[ind]

    if max_rows is not None:
        _stats = stats[:max_rows]
    else:
        _stats = stats

    datas = [
        (
            fn.name,
            _path_to_module(fn.module),
            f"{fn.ncall}",
            fn.ttot,
            fn.tsub,
            fn.tavg,
            fn.ttot / total * 100,
        )
        for fn in _stats
    ]
    for data in datas:
        for ind in range(3):
            if column_sizes[ind] < len(data[ind]):
                column_sizes[ind] = len(data[ind])

    _str = " | ".join([f"{title:^{size}}" for title, size in zip(header, column_sizes)])
    print(_str)
    print("-" * len(_str))

    for data in datas:
        _str = " | ".join(
            [f"{x:{fmt}}".ljust(size) for x, size, fmt in zip(data[: len(header)], column_sizes, formats)]
        )
        print(_str)


@check_yappi
def profile_clear() -> None:
    global START_TIME
    START_TIME = time.time()
    yappi.clear_stats()


@check_yappi
def profile_stop(clear: bool = True) -> None:
    yappi.stop()
    if clear:
        global START_TIME
        START_TIME = 0.0
        yappi.clear_stats()
