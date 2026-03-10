import logging
from typing import Optional

from hardtarget.constants import EventDetectionMethod, Impl
from hardtarget.types import EventSearchLib, MethodLib

logger = logging.getLogger(__name__)

# available libs
EVENT_DETECTION_LIBS: dict[EventDetectionMethod, dict[Impl, EventSearchLib]] = {
    method: {} for method in EventDetectionMethod
}


try:
    from .xcorr.xcorr_c import xcorr_c
except ImportError as err:
    logger.debug(f"Event search c implementations failed to import:\n {err}", exc_info=True)
else:
    EVENT_DETECTION_LIBS[EventDetectionMethod.xcorr][Impl.c] = xcorr_c


def get_event_detection_lib(
    method_lib: Optional[MethodLib] = None, implementation: Optional[Impl] = None
) -> tuple[EventSearchLib, EventDetectionMethod, Impl]:

    if method_lib is None:
        logger.debug("No method defined, default lib will be used")
        method_lib = EventDetectionMethod.xcorr

    if implementation is None:
        logger.debug("No implementation defined, default implementation will be used")
        implementation = Impl.c

    if not isinstance(method_lib, EventDetectionMethod):
        try:
            method_lib = EventDetectionMethod(method_lib)
        except ValueError:
            raise Exception(f"Method: {method_lib} is not available for event detection")

    lib = EVENT_DETECTION_LIBS[method_lib].get(implementation, None)
    if lib is None:
        raise Exception(f"There is no available {implementation} implementation for method lib {method_lib}")

    return lib, method_lib, implementation


from .types import XCorrCfgParams, XCorrOutArgs, XCorrProParams, XCorrVariables
from .xcorr_process import XCorrProcess
