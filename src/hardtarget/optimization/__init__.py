import logging
from typing import Optional

from hardtarget.constants import Impl, OptimizationMethod
from hardtarget.types import MethodLib, OptimizeLib

logger = logging.getLogger(__name__)

# available libs
OPTIMIZE_LIBS: dict[OptimizationMethod, dict[Impl, OptimizeLib]] = {
    method: {} for method in OptimizationMethod
}


try:
    from .optimize_gmf_numpy import optimize_gmf_np, optimize_grid_gmf_np
except ImportError as err:
    logger.debug(f"Optimize Numpy implementations failed to import:\n {err}", exc_info=True)
else:
    OPTIMIZE_LIBS[OptimizationMethod.optimize_scipy_gmf][Impl.numpy] = optimize_gmf_np
    OPTIMIZE_LIBS[OptimizationMethod.optimize_grid_gmf][Impl.numpy] = optimize_grid_gmf_np


def get_optimize_lib(
    method_lib: Optional[MethodLib] = None, implementation: Optional[Impl] = None
) -> tuple[OptimizeLib, OptimizationMethod, Impl]:

    if method_lib is None:
        logger.debug("No method defined, default lib will be used")
        method_lib = OptimizationMethod.optimize_grid_gmf

    if implementation is None:
        logger.debug("No implementation defined, default implementation will be used")
        implementation = Impl.numpy

    if not isinstance(method_lib, OptimizationMethod):
        try:
            method_lib = OptimizationMethod(method_lib)
        except ValueError:
            raise Exception(f"Method: {method_lib} is not available for optimization")

    lib = OPTIMIZE_LIBS[method_lib].get(implementation, None)
    if lib is None:
        raise Exception(f"There is no available {implementation} implementation for method lib {method_lib}")

    return lib, method_lib, implementation


from .optimize_process import OptimizeProcess
from .types import MFOptimizeOutArgs, MFOptimizeVariables, OptimizeCfgParams, OptimizeProParams
