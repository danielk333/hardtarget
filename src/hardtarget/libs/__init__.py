import ctypes
import pathlib
import sysconfig

import numpy as np

from .numpy_ctypes import nptype


def load_c_lib() -> ctypes.CDLL:
    """Load the C library"""

    # Load the C-lib
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        suffix = ".so"

    # We start by making a path to the current directory.
    pymodule_dir = pathlib.Path(__file__).resolve().parent
    __libpath__ = pymodule_dir / ("clib" + suffix)

    if __libpath__.is_file():
        # Then we open the created shared lib file
        clib = ctypes.cdll.LoadLibrary(str(__libpath__))

        # https://docs.python.org/3/library/ctypes.html#ctypes._CFuncPtr.argtypes

        clib.fgmf.restype = ctypes.c_int
        clib.fgmf.argtypes = [
            nptype(np.complex64, 1),  # 1
            ctypes.c_int,  # 2
            nptype(np.complex64, 1),  # 3
            ctypes.c_int,  # 4
            ctypes.c_int,  # 5
            nptype(np.complex64, 2),  # 6
            ctypes.c_int,  # 7
            nptype(np.int32, 1),  # 8
            ctypes.c_int,  # 9
            ctypes.c_int,  # 10
            nptype(np.float32, 1, w=True),  # 11
            nptype(np.float32, 1, w=True),  # 12
            nptype(np.int32, 1, w=True),  # 13
            nptype(np.int32, 1, w=True),  # 14
            nptype(np.int32, 1),  # 15
            nptype(np.int32, 1),  # 16
            ctypes.c_int,  # 17
        ]

        clib.fdpt.restype = ctypes.c_int
        clib.fdpt.argtypes = [
            nptype(np.complex64, 1),  # 1
            ctypes.c_int,  #  2
            nptype(np.complex64, 1),  # 3
            ctypes.c_int,  # 4
            ctypes.c_int,  # 5
            nptype(np.complex64, 2),  # 6
            ctypes.c_int,  # 7
            nptype(np.int32, 1),  # 8
            ctypes.c_int,  # 9
            ctypes.c_int,  # 10
            nptype(np.float32, 1, w=True),  # 11
            nptype(np.float32, 1, w=True),  # 12
            nptype(np.int32, 1, w=True),  # 13
            nptype(np.int32, 1, w=True),  # 14
            nptype(np.int32, 1),  # 15
            nptype(np.int32, 1),  # 16
            ctypes.c_int,  # 17
            ctypes.c_int,  # 18
        ]

        clib.xcorr_echo_search.restype = ctypes.c_int
        clib.xcorr_echo_search.argtypes = [
            nptype(np.complex64, 1),  # 1
            ctypes.c_int,  #  2
            nptype(np.complex64, 1),  # 3
            ctypes.c_int,  # 4
            ctypes.c_int,  # 5
            ctypes.c_int,  # 6
            ctypes.c_int,  # 7
            ctypes.c_int,  # 8
            ctypes.c_int,  # 9
            nptype(np.complex64, 2, w=True),  # 10
            nptype(np.int32, 1, w=True),  # 11
            nptype(np.complex64, 2, w=True),  # 12
            nptype(np.int32, 1, w=True),  # 13
            nptype(np.complex64, 1, w=True),  # 14
            ctypes.c_int,  # 15
            nptype(np.int32, 1, w=True),  # 16
            ctypes.c_int,  # 17
        ]

    else:
        raise ImportError(f"{__libpath__} GMF C Library not found")

    return clib


def load_cuda_lib() -> ctypes.CDLL:
    """Load the Cuda library"""

    # Load the C-lib
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        suffix = ".so"

    # We start by making a path to the current directory.
    pymodule_dir = pathlib.Path(__file__).resolve().parent
    __libpath__ = pymodule_dir / ("cudalib" + suffix)

    if __libpath__.is_file():
        # Then we open the created shared lib file
        cudalib = ctypes.cdll.LoadLibrary(str(__libpath__))

        cudalib.gmf.restype = ctypes.c_int
        cudalib.gmf.argtypes = [
            nptype(np.complex64, 1),  # 1
            ctypes.c_int,  #  2
            nptype(np.complex64, 1),  # 3
            ctypes.c_int,  # 4
            nptype(np.complex64, 2),  # 5
            ctypes.c_int,  #  6
            nptype(np.int32, 1),  # 7
            ctypes.c_int,  #  8
            ctypes.c_int,  # 9
            nptype(np.float32, 1, w=True),  # 10
            nptype(np.float32, 1, w=True),  # 11
            nptype(np.int32, 1, w=True),  # 12
            nptype(np.int32, 1, w=True),  # 13
            nptype(np.int32, 1),  # 14
            nptype(np.int32, 1),  # 15
            ctypes.c_int,  # 16
            ctypes.c_int,  # 17
        ]

        cudalib.print_devices.restype = None
        cudalib.print_devices.argtypes = []

        cudalib.test_cuda.restype = ctypes.c_int
        cudalib.test_cuda.argtypes = []
    else:
        raise ImportError(f"{__libpath__} GMF Cuda Library not found")

    return cudalib
