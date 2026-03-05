"""
GMF estimation implementations
===============================

This subpackage contains the different available implementations that calculate
or approximate the Generalized Matched Filter (GMF). The GMF for a certain
signal model is proportional to the Likelihood function for that signal models
parameters given a measured signal. It is called a matched filter because signal
power is transmitted trough the filter (i.e. the function) where the model
matches the recorded signal. If the measured signal follows the signal model,
the peak of the GMF appears at the location of the parameters of the true signal
perturbed by noise. For multiple targets it is usually possible to find multiple
peaks in the GMF each corresponding the a unique target.


Estimation methods
------------------

Currently there are three different ways to calculate or approximate global
maximum of the GMF. These methods are divided into two categories, grid based
methods and optimization based methods.

Grid methods:

- Fast discrete polynomial-phase transform (FDPT)
- Fast-GMF (FGMF)

Optimize methods:

- Maximum Likelihood (GMF)

The real GMF function is too expensive to calculate on a grid and instead uses
some optimization to calculate the peak value. Since the GMF is often quite
"bumpy" and not unimodal, this method need a good initial guess. Hence the usual
workflow is to first run a grid method and then run a optimization method using
the maximum of the grid as a seed value.

Implementations Language
------------------------

These different methods can be implemented in different programming languages
and on different hardware. Currently there are three main implementation methods
that span both the CPU and GPU:

- numpy
- c
- cuda

"""

import logging
from typing import Callable

from hardtarget.types.constants import AnalysisMethod, EstimationMethod, Impl
from hardtarget.types.types import AnalysisLib, OptimizeLib

from . import types, utils

logger = logging.getLogger(__name__)


# --- Libraries ----
ANALYSIS_LIBS: dict[Impl, dict[EstimationMethod, tuple[AnalysisLib | OptimizeLib, AnalysisMethod]]] = {
    imp: {} for imp in Impl
}

# Numpy implementation
try:
    from .gmf.gmf_numpy import fast_gmf_no_reduce_np, fast_gmf_np
except ImportError as err:
    logger.debug(f"GMF Numpy implementations failed to import:\n {err}", exc_info=True)
else:
    ANALYSIS_LIBS[Impl.numpy][EstimationMethod.grid_fast_gmf] = (fast_gmf_np, AnalysisMethod.gmf)
    ANALYSIS_LIBS[Impl.numpy][EstimationMethod.grid_fast_no_reduce] = (
        fast_gmf_no_reduce_np,
        AnalysisMethod.gmf,
    )
    # Alias
    ANALYSIS_LIBS[Impl.numpy][EstimationMethod.fgmf] = ANALYSIS_LIBS[Impl.numpy][
        EstimationMethod.grid_fast_gmf
    ]

try:
    from .dpt.dpt_numpy import fast_dpt_np
except ImportError as err:
    logger.debug(f"DPT Numpy implementations failed to import:\n {err}", exc_info=True)
else:
    ANALYSIS_LIBS[Impl.numpy][EstimationMethod.grid_fast_dpt] = (fast_dpt_np, AnalysisMethod.dpt)
    # Alias
    ANALYSIS_LIBS[Impl.numpy][EstimationMethod.fdpt] = ANALYSIS_LIBS[Impl.numpy][
        EstimationMethod.grid_fast_dpt
    ]

try:
    from .optimize.optimize_gmf_numpy import optimize_gmf_np, optimize_grid_gmf_np
except ImportError as err:
    logger.debug(f"Optimize Numpy implementations failed to import:\n {err}", exc_info=True)
else:
    ANALYSIS_LIBS[Impl.numpy][EstimationMethod.optimize_scipy_gmf] = (
        optimize_gmf_np,
        AnalysisMethod.optimize,
    )
    ANALYSIS_LIBS[Impl.numpy][EstimationMethod.optimize_grid_gmf] = (
        optimize_grid_gmf_np,
        AnalysisMethod.optimize,
    )

# C implementation
try:
    from .gmf.gmf_c import fast_gmf_c
except ImportError as err:
    logger.debug(f"GMF c implementations failed to import:\n {err}", exc_info=True)
else:
    ANALYSIS_LIBS[Impl.c][EstimationMethod.grid_fast_gmf] = (fast_gmf_c, AnalysisMethod.gmf)
    ANALYSIS_LIBS[Impl.c][EstimationMethod.fgmf] = ANALYSIS_LIBS[Impl.c][EstimationMethod.grid_fast_gmf]

try:
    from .dpt.dpt_c import fast_dpt_c
except ImportError as err:
    logger.debug(f"DPT c implementations failed to import:\n {err}", exc_info=True)
else:
    ANALYSIS_LIBS[Impl.c][EstimationMethod.grid_fast_dpt] = (fast_dpt_c, AnalysisMethod.dpt)
    ANALYSIS_LIBS[Impl.c][EstimationMethod.fdpt] = ANALYSIS_LIBS[Impl.c][EstimationMethod.grid_fast_dpt]

# Cuda implementation
try:
    from .gmf.gmf_cuda import fast_gmf_cuda
except ImportError as err:
    logger.debug(f"GMF cuda implementations failed to import:\n {err}", exc_info=True)
else:
    ANALYSIS_LIBS[Impl.cuda][EstimationMethod.grid_fast_gmf] = (fast_gmf_cuda, AnalysisMethod.gmf)
    ANALYSIS_LIBS[Impl.cuda][EstimationMethod.fgmf] = ANALYSIS_LIBS[Impl.cuda][EstimationMethod.grid_fast_gmf]


def get_available_libs(indent: str = "") -> str:
    st = ""
    for imp in Impl:
        st += indent + f"[{imp.value}]:\n"
        if len(ANALYSIS_LIBS[imp]) == 0:
            st += "[No implementations]\n"
            continue

        max_name_len = max([len(name) for name in ANALYSIS_LIBS[imp]])
        for name, (func, mtype) in ANALYSIS_LIBS[imp].items():
            st += indent + f" - {name.ljust(max_name_len, ' ')} ({mtype.value} method)\n"
    return st


def get_estimation_method(
    implementation: Impl, name: EstimationMethod
) -> tuple[OptimizeLib | AnalysisLib | None, AnalysisMethod]:
    """Get implementation method by name.

    Returns tuple with function pointer and method type
    """

    return ANALYSIS_LIBS[implementation].get(name, (None, AnalysisMethod.unknown))


def get_default_method() -> tuple[Impl, EstimationMethod]:
    """Default estimation method"""
    imp = Impl.c
    lib, libtype = get_estimation_method(imp, EstimationMethod.grid_fast_dpt)
    # Fallbacks
    if lib is None:
        imp = Impl.numpy
        lib, libtype = get_estimation_method(imp, EstimationMethod.grid_fast_dpt)

    if lib is None:
        imp = Impl.unknown
        logger.warning("No default estimation method set.")

    return imp, EstimationMethod.fgmf


# isort: off
from hardtarget.process import Process
from .dpt.dpt_process import DPTProcess
from .gmf.gmf_process import GMFProcess
from .optimize.optimize_process import OptimizeProcess

# isort: on

# ---- Processes ----
PROCESSES: dict[AnalysisMethod, Callable[..., Process]] = {
    AnalysisMethod.gmf: GMFProcess,
    AnalysisMethod.dpt: DPTProcess,
    AnalysisMethod.optimize: OptimizeProcess,
}


def get_analysis_process(method: AnalysisMethod) -> Callable[..., Process]:
    return PROCESSES[method]
