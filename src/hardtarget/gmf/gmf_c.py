import ctypes
import sysconfig
import pathlib

# Load the C-lib
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

# We start by making a path to the current directory.
pymodule_dir = pathlib.Path(__file__).resolve().parent
__libpath__ = pymodule_dir / ('gmfclib' + suffix)

if __libpath__.is_file():
    # Then we open the created shared lib file
    gmfclib = ctypes.cdll.LoadLibrary(__libpath__)

    gmfclib.gmf.restype = ctypes.c_int
    gmfclib.gmf.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
    ]
else:
    raise ImportError(f'{__libpath__} GMF C Library not found')


def gmf_c(z_tx, z_rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec):
    # TODO: This should be pythonified
    error_code = gmfclib.gmf(
        z_tx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        z_tx.size,
        z_rx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        z_rx.size,
        acc_phasors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acc_phasors.shape[0],
        rgs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(rgs),
        dec,
        gmf_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gmf_dc_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        a_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
    )
    assert error_code == 0, f"GMF C-function returned error {error_code}"
