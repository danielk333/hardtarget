"""Project constants"""

# Python StrEnum has default lowercase for auto() but is only available from py 3.11
try:
    from enum import StrEnum
except ImportError:
    from strenum import (  # type: ignore[assignment, no-redef, unused-ignore, import-not-found]
        LowercaseStrEnum as StrEnum,  # type: ignore[import-not-found,no-redef, unused-ignore]
    )


class AnalysisMethod(StrEnum):
    """Analysis method type"""

    gmf = "gmf"
    dpt = "dpt"
    optimize = "optimize"
    unknown = "unknown"


class Impl(StrEnum):
    """Implementation language"""

    numpy = "numpy"
    c = "c"
    cuda = "cuda"
    unknown = "unknown"


class GMFMethod(StrEnum):
    """GMF Analysis Methods"""

    grid_fast_gmf = "grid-fast-gmf"
    fgmf = "fgmf"
    grid_fast_no_reduce = "grid_fast_no_reduce"
    unknown = "unknown"


class DPTMethod(StrEnum):
    """DPT Analysis Methods"""

    grid_fast_dpt = "grid-fast-dpt"
    fdpt = "fdpt"
    unknown = "unknown"


class OptimizationMethod(StrEnum):
    """GMF Optimization Methods"""

    optimize_scipy_gmf = "optimize-scipy-gmf"
    optimize_grid_gmf = "optimize-grid-gmf"
    unknown = "unknown"


class EstimationMethod(StrEnum):
    """All estimation methods"""

    grid_fast_gmf = "grid-fast-gmf"
    grid_fast_dpt = "grid-fast-dpt"
    fgmf = "fgmf"
    fdpt = "fdpt"
    grid_fast_no_reduce = "grid_fast_no_reduce"
    optimize_scipy_gmf = "optimize-scipy-gmf"
    optimize_grid_gmf = "optimize-grid-gmf"
    unknown = "unknown"


class ConfigSubSection(StrEnum):
    """Subsections in configuration .ini file"""

    PROCCESSING = "processing"
    GMF = "gmf"
    DPT = "dpt"
    OPTIMIZATION = "optimization"
    INFOMETRY = "infometry"
