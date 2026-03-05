import ctypes
import pathlib
import sysconfig

from .numpy_ctypes import nptype
from .utils import default_mf_vars_items


def load_c_lib() -> ctypes.CDLL:
    """Load the C library"""

    # Load the C-lib
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        suffix = ".so"

    # We start by making a path to the current directory.
    pymodule_dir = pathlib.Path(__file__).resolve().parent
    __libpath__ = pymodule_dir / ("gmfclib" + suffix)

    if __libpath__.is_file():
        # Then we open the created shared lib file
        gmfclib = ctypes.cdll.LoadLibrary(str(__libpath__))

        gmfclib.gmf.restype = ctypes.c_int
        # TODO: rename these to better names
        # - probably gmf -> fast-gmf
        # full amplitude domain function -> gmf
        # discrete polynomial phase transform -> fast-dpt
        # https://docs.python.org/3/library/ctypes.html#ctypes._CFuncPtr.argtypes
        gmfclib.gmf.argtypes = [
            nptype("c8", 1),  # 1
            ctypes.c_int,  # 2
            nptype("c8", 1),  # 3
            ctypes.c_int,  # 4
            ctypes.c_int,  # 5
            nptype("c8", 2),  # 6
            ctypes.c_int,  # 7
            nptype("i4", 1),  # 8
            ctypes.c_int,  # 9
            ctypes.c_int,  # 10
            nptype("f4", 1, w=True),  # 11
            nptype("f4", 1, w=True),  # 12
            nptype("i4", 1, w=True),  # 13
            nptype("i4", 1, w=True),  # 14
            nptype("i4", 1),  # 15
            nptype("i4", 1),  # 16
            ctypes.c_int,  # 17
        ]

        gmfclib.dpt.restype = ctypes.c_int
        gmfclib.dpt.argtypes = [
            nptype("c8", 1),  # 1
            ctypes.c_int,  #  2
            nptype("c8", 1),  # 3
            ctypes.c_int,  # 4
            ctypes.c_int,  # 5
            nptype("c8", 2),  # 6
            ctypes.c_int,  # 7
            nptype("i4", 1),  # 8
            ctypes.c_int,  # 9
            ctypes.c_int,  # 10
            nptype("f4", 1, w=True),  # 11
            nptype("f4", 1, w=True),  # 12
            nptype("i4", 1, w=True),  # 13
            nptype("i4", 1, w=True),  # 14
            nptype("i4", 1),  # 15
            nptype("i4", 1),  # 16
            ctypes.c_int,  # 17
            ctypes.c_int,  # 18
        ]
    else:
        raise ImportError(f"{__libpath__} GMF C Library not found")

    return gmfclib


def load_cuda_lib() -> ctypes.CDLL:
    """Load the Cuda library"""

    # Load the C-lib
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        suffix = ".so"

    # We start by making a path to the current directory.
    pymodule_dir = pathlib.Path(__file__).resolve().parent
    __libpath__ = pymodule_dir / ("gmfcudalib" + suffix)

    if __libpath__.is_file():
        # Then we open the created shared lib file
        gmfcudalib = ctypes.cdll.LoadLibrary(str(__libpath__))

        gmfcudalib.gmf.restype = ctypes.c_int
        gmfcudalib.gmf.argtypes = [
            nptype("c8", 1),
            ctypes.c_int,  # 1, 2
            nptype("c8", 1),
            ctypes.c_int,  # 3, 4
            nptype("c8", 2),
            ctypes.c_int,  # 5, 6
            nptype("i4", 1),
            ctypes.c_int,  # 7, 8
            ctypes.c_int,  # 9
            nptype("f4", 1, w=True),  # 10
            nptype("f4", 1, w=True),  # 11
            nptype("i4", 1, w=True),  # 12
            nptype("i4", 1, w=True),  # 13
            nptype("i4", 1),  # 14
            nptype("i4", 1),  # 15
            ctypes.c_int,  # 16
            ctypes.c_int,  # 17
        ]

        gmfcudalib.print_devices.restype = None
        gmfcudalib.print_devices.argtypes = []

        gmfcudalib.test_cuda.restype = ctypes.c_int
        gmfcudalib.test_cuda.argtypes = []
    else:
        raise ImportError(f"{__libpath__} GMF Cuda Library not found")

    return gmfcudalib
