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
from enum import Enum

from .gmf_numpy import (
    fast_gmf_np,
    fast_dpt_np,
    fast_gmf_no_reduce_np,
    optimize_gmf_np,
)

logger = logging.getLogger(__name__)


class MethodType(Enum):
    """Method type"""

    grid = "grid"
    optimize = "optimize"


class Impl(Enum):
    """Implementation language"""

    numpy = "numpy"
    c = "c"
    cuda = "cuda"


GMF_LIBS = {imp: {} for imp in Impl}

GMF_LIBS[Impl.numpy]["fgmf"] = (fast_gmf_np, MethodType.grid)
GMF_LIBS[Impl.numpy]["fdpt"] = (fast_dpt_np, MethodType.grid)
GMF_LIBS[Impl.numpy]["fgmf_no_reduce"] = (fast_gmf_no_reduce_np, MethodType.grid)
GMF_LIBS[Impl.numpy]["gmf"] = (optimize_gmf_np, MethodType.optimize)


try:
    from .gmf_c import gmfc
except ImportError as err:
    logger.debug(f"GMF c implementations failed to import:\n {e}", exc_info=True)
else:
    GMF_LIBS[Impl.c]["fgmf"] = (gmfc, MethodType.grid)

try:
    from .gmf_cuda import gmfcu
except ImportError as err:
    logger.debug(f"GMF cuda implementations failed to import:\n {e}", exc_info=True)
else:
    GMF_LIBS[Impl.cuda]["fgmf"] = (gmfcu, MethodType.grid)


def get_avalible_libs(indent=""):
    st = ""
    for imp in Impl:
        st += indent + f"[{imp.value}]:\n"
        max_name_len = max([len(name) for name in GMF_LIBS[imp]])
        for name, (func, mtype) in GMF_LIBS[imp].items():
            st += indent + f" - {name.ljust(max_name_len, ' ')} ({mtype.value} method)\n"
    return st


def get_estimation_method(implemenation, name):
    """Get implementation method by name.
    
    Returns tuple with function pointer and method type
    """
    return GMF_LIBS[Impl(implemenation)].get(name, (None, None))


def get_default_method():
    """Default estimation method
    """
    imp = "c"
    lib, libtype = get_estimation_method(imp, "fdpt")
    # Fallbacks
    if lib is None:
        imp = "numpy"
        lib, libtype = get_estimation_method(imp, "fdpt")

    if lib is None:
        imp = None
        logger.warning("No default estimation method set.")

    return imp, lib
