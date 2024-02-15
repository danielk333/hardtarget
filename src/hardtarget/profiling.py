import logging
import sys
import pathlib
import datetime
import re

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

logger = logging.getLogger(__name__)


def get_logging_level(verbosity):
    if verbosity <= 0:
        return logging.WARNING
    elif verbosity == 1:
        return logging.INFO
    elif verbosity >= 2:
        return logging.DEBUG
    else:
        raise ValueError(f"Verbosity '{verbosity}' not recognized")


def check_parallel():
    if comm is None:
        return False, None
    else:
        if comm.size > 1:
            return True, comm.rank
        else:
            return False, None


def modify_subloggers(matching_regex):
    raise NotImplementedError()
    _ = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if re.search(matching_regex, name) is not None
    ]


def setup_loggers(log_folder=None, stdout=True, verbosity=0):
    parallel, rank = check_parallel()
    if parallel:
        formatter = logging.Formatter(
            f"RANK{rank} - %(asctime)s %(levelname)s %(name)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
    formatter.default_msec_format = "%s.%03d"

    handlers = []
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
