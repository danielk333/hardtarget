import logging
from typing import Optional

from hardtarget.constants import Impl, TargetEstimationMethod
from hardtarget.types import TargetEstimationLib, MethodLib

logger = logging.getLogger(__name__)


# available libs
DPT_LIBS: dict[TargetEstimationMethod, dict[Impl, TargetEstimationLib]] = {
    method: {} for method in TargetEstimationMethod
}


try:
    from .dpt_c import fast_dpt_c
except ImportError as err:
    logger.debug(f"DPT c implementations failed to import:\n {err}", exc_info=True)
else:
    DPT_LIBS[TargetEstimationMethod.fdpt][Impl.c] = fast_dpt_c


try:
    from .dpt_numpy import fast_dpt_np
except ImportError as err:
    logger.debug(f"DPT Numpy implementations failed to import:\n {err}", exc_info=True)
else:
    DPT_LIBS[TargetEstimationMethod.fdpt][Impl.numpy] = fast_dpt_np


def get_dbt_lib(
    method_lib: Optional[MethodLib] = None, implementation: Optional[Impl] = None
) -> tuple[TargetEstimationLib, TargetEstimationMethod, Impl]:

    if method_lib is None:
        logger.debug("No method defined, default lib will be used")
        method_lib = TargetEstimationMethod.fdpt

    if implementation is None:
        logger.debug("No implementation defined, default implementation will be used")
        implementation = Impl.numpy

    if not isinstance(method_lib, TargetEstimationMethod):
        try:
            method_lib = TargetEstimationMethod(method_lib)
        except ValueError:
            raise Exception(f"Method: {method_lib} is not available for target estimation")

    lib = DPT_LIBS[method_lib].get(implementation, None)
    if lib is None:
        raise Exception(f"Ther is no available {implementation} implementation for method lib {method_lib}")

    return lib, method_lib, implementation


from .dpt_process import DPTProcess
from .types import DPTCfgParams, DPTProParams
