"""Project constants"""

from typing import TypeAlias

# Python StrEnum has default lowercase for auto() but is only available from py 3.11
try:
    from enum import StrEnum
except ImportError:
    from strenum import (  # type: ignore[assignment, no-redef, unused-ignore, import-not-found]
        LowercaseStrEnum as StrEnum,  # type: ignore[import-not-found,no-redef, unused-ignore]
    )


class AnalysisMethod(StrEnum):
    """Analysis method type"""

    target_estimation = "target_estimation"
    optimize = "optimize"
    event_detection = "event_detection"
    direction_of_arrival = "direction_of_arrival"
    unknown = "unknown"


class TargetEstimationMethod(StrEnum):
    """Target estimation methods"""

    fgmf = "fgmf"
    fdpt = "fdpt"
    grid_fast_no_reduce = "grid_fast_no_reduce"
    unknown = "unknown"


class OptimizationMethod(StrEnum):
    """GMF Optimization Methods"""

    optimize_scipy_gmf = "optimize-scipy-gmf"
    optimize_grid_gmf = "optimize-grid-gmf"


class EventDetectionMethod(StrEnum):
    """Event detection methods"""

    xcorr = "xcorr"


class DOAMethod(StrEnum):
    """Direction of arrival methods"""

    music_grid_search = "music_grid_search"
    beamforming_grid_search = "beamforming_grid_search"


MethodLib: TypeAlias = TargetEstimationMethod | OptimizationMethod | EventDetectionMethod | DOAMethod


class Impl(StrEnum):
    """Implementation language"""

    numpy = "numpy"
    c = "c"
    cuda = "cuda"
    unknown = "unknown"


class Processes(StrEnum):
    GMF = "gmf"
    DPT = "dpt"
    Optimization = "optimization"
    XCORR = "xcorr"
    DOA = "doa"


class ConfigSubSection(StrEnum):
    """Subsections in configuration .ini file"""

    PROCCESSING = "processing"
    GMF = "gmf"
    DPT = "dpt"
    OPTIMIZATION = "optimization"
    INTERFEROMETRY = "interferometry"
    XCORR = "xcorr"
