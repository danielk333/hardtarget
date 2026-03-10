import logging
from typing import Optional

from hardtarget.constants import DOAMethod, Impl
from hardtarget.types import InterferometryLib, MethodLib

logger = logging.getLogger(__name__)


# available libs
DOA_LIBS: dict[DOAMethod, dict[Impl, InterferometryLib]] = {method: {} for method in DOAMethod}

try:
    from .music import music_numpy
except ImportError as err:
    logger.debug(f"MUSIC Numpy implementations failed to import:\n {err}", exc_info=True)
else:
    DOA_LIBS[DOAMethod.music_grid_search][Impl.numpy] = music_numpy.grid_search_numpy

try:
    from .beamforming import bf_numpy
except ImportError as err:
    logger.debug(f"Beamforming numpy implementations failed to import:\n {err}", exc_info=True)
else:
    DOA_LIBS[DOAMethod.beamforming_grid_search][Impl.numpy] = bf_numpy.grid_search_numpy


def get_doa_lib(
    method_lib: Optional[MethodLib] = None, implementation: Optional[Impl] = None
) -> tuple[InterferometryLib, DOAMethod, Impl]:

    if method_lib is None:
        logger.debug("No method defined, default lib will be used")
        method_lib = DOAMethod.music_grid_search

    if implementation is None:
        logger.debug("No implementation defined, default implementation will be used")
        implementation = Impl.numpy

    if not isinstance(method_lib, DOAMethod):
        try:
            method_lib = DOAMethod(method_lib)
        except ValueError:
            raise Exception(f"Method: {method_lib} is not available for direction of arrival")

    lib = DOA_LIBS[method_lib].get(implementation, None)
    if lib is None:
        raise Exception(f"Ther is no available {implementation} implementation for method lib {method_lib}")

    return lib, method_lib, implementation


from .doa_process import DOAProcess
from .types import DOACfgParams, DOAProParams, DOAVars
