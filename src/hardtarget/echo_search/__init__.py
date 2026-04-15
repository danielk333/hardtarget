import logging
from typing import Optional

from hardtarget.constants import EchoSearchMethod, Impl
from hardtarget.types import EventSearchLib, MethodLib

logger = logging.getLogger(__name__)

# available libs
ECHO_SEARCH_LIBS: dict[EchoSearchMethod, dict[Impl, EventSearchLib]] = {
    method: {} for method in EchoSearchMethod
}


try:
    from .xcorr.xcorr_c import xcorr_c
except ImportError as err:
    logger.debug(f"Event search c implementations failed to import:\n {err}", exc_info=True)
else:
    ECHO_SEARCH_LIBS[EchoSearchMethod.xcorr][Impl.c] = xcorr_c


def get_echo_search_lib(
    method_lib: Optional[MethodLib] = None, implementation: Optional[Impl] = None
) -> tuple[EventSearchLib, EchoSearchMethod, Impl]:

    if method_lib is None:
        logger.debug("No method defined, default lib will be used")
        method_lib = EchoSearchMethod.xcorr

    if implementation is None:
        logger.debug("No implementation defined, default implementation will be used")
        implementation = Impl.c

    if not isinstance(method_lib, EchoSearchMethod):
        try:
            method_lib = EchoSearchMethod(method_lib)
        except ValueError:
            raise Exception(f"Method: {method_lib} is not available for echo search")

    lib = ECHO_SEARCH_LIBS[method_lib].get(implementation, None)
    if lib is None:
        raise Exception(f"There is no available {implementation} implementation for method lib {method_lib}")

    return lib, method_lib, implementation


from .echo_search_process import EchoSearchProcess
from .types import EchoSearchCfgParams, EchoSearchOutArgs, EchoSearchProParams, EchoSearchVars
