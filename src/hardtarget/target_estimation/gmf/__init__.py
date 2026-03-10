import logging
from typing import Optional

from hardtarget.constants import Impl, TargetEstimationMethod
from hardtarget.types import AnalysisLib, MethodLib

logger = logging.getLogger(__name__)

# available libs
GMF_LIBS: dict[TargetEstimationMethod, dict[Impl, AnalysisLib]] = {
    method: {} for method in TargetEstimationMethod
}

# Numpy implementation
try:
    from .gmf_numpy import fast_gmf_no_reduce_np, fast_gmf_np
except ImportError as err:
    logger.debug(f"GMF Numpy implementations failed to import:\n {err}", exc_info=True)
else:
    GMF_LIBS[TargetEstimationMethod.fgmf][Impl.numpy] = fast_gmf_np
    GMF_LIBS[TargetEstimationMethod.grid_fast_no_reduce][Impl.numpy] = fast_gmf_no_reduce_np
# C implementation
try:
    from .gmf_c import fast_gmf_c
except ImportError as err:
    logger.debug(f"GMF c implementations failed to import:\n {err}", exc_info=True)
else:
    GMF_LIBS[TargetEstimationMethod.fgmf][Impl.c] = fast_gmf_c

# Cuda implementation
try:
    from .gmf_cuda import fast_gmf_cuda
except ImportError as err:
    logger.debug(f"GMF cuda implementations failed to import:\n {err}", exc_info=True)
else:
    GMF_LIBS[TargetEstimationMethod.fgmf][Impl.cuda] = fast_gmf_cuda


def get_gmf_lib(
    method_lib: Optional[MethodLib] = None, implementation: Optional[Impl] = None
) -> tuple[AnalysisLib, TargetEstimationMethod, Impl]:

    if method_lib is None:
        logger.debug("No method defined, default lib will be used")
        method_lib = TargetEstimationMethod.fgmf

    if implementation is None:
        logger.debug("No implementation defined, default implementation will be used")
        implementation = Impl.numpy

    if not isinstance(method_lib, TargetEstimationMethod):
        try:
            method_lib = TargetEstimationMethod(method_lib)
        except ValueError:
            raise Exception(f"Method: {method_lib} is not available for target estimation")

    lib = GMF_LIBS[method_lib].get(implementation, None)
    if lib is None:
        raise Exception(f"There is no available {implementation} implementation for method lib {method_lib}")

    return lib, method_lib, implementation


from .gmf_process import GMFProcess
from .types import GMFCfgParams, GMFProParams
