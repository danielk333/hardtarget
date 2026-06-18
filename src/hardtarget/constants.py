"""Project constants"""

from typing import TypeAlias

try:
    # Python StrEnum has default lowercase for auto() but is only available from py 3.11
    from enum import StrEnum
except ImportError:
    from strenum import (  # type: ignore[assignment, no-redef, unused-ignore, import-not-found]
        LowercaseStrEnum as StrEnum,  # type: ignore[import-not-found,no-redef, unused-ignore]
    )


class FIRFilter(StrEnum):
    """FIR Filter type"""

    b414d15_gaus = "b414d15_gaus"
    unknown = "unknown"


class AnalysisMethod(StrEnum):
    """Analysis method type"""

    target_estimation = "target_estimation"
    optimize = "optimize"
    echo_search = "echo_search"
    direction_of_arrival = "direction_of_arrival"
    unknown = "unknown"


class MethodAbbreviation(StrEnum):
    target_estimation = "te"
    optimize = "opt"
    echo_search = "echo"
    direction_of_arrival = "doa"


class TargetEstimationMethod(StrEnum):
    """Target estimation methods"""

    fdpt = "fdpt"
    fgmf = "fgmf"
    # grid_fast_no_reduce = "grid_fast_no_reduce"


class OptimizationMethod(StrEnum):
    """GMF Optimization Methods"""

    optimize_scipy_gmf = "optimize_scipy_gmf"
    optimize_grid_gmf = "optimize_grid_gmf"


class EchoSearchMethod(StrEnum):
    """Echo search methods"""

    xcorr = "xcorr"


class DOAMethod(StrEnum):
    """Direction of arrival methods"""

    music_grid_search = "music_grid_search"
    beamforming_grid_search = "beamforming_grid_search"


MethodLib: TypeAlias = TargetEstimationMethod | OptimizationMethod | EchoSearchMethod | DOAMethod


class Impl(StrEnum):
    """Implementation language"""

    numpy = "numpy"
    c = "c"
    cuda = "cuda"
    unknown = "unknown"


class ConfigSubSection(StrEnum):
    """Subsections in configuration .ini file"""

    PROCCESSING = "processing"
    TARGET_ESTIMATION = "target_estimation"
    GMF = "gmf"
    DPT = "dpt"
    OPTIMIZATION = "optimization"
    INTERFEROMETRY = "interferometry"
    ECHO_SEARCH = "echo_search"
