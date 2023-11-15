import ctypes
import sysconfig
import pathlib

# Load the C-lib
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

# We start by making a path to the current directory.
pymodule_dir = pathlib.Path(__file__).resolve().parent
__libpath__ = pymodule_dir / ('gmfcudalib' + suffix)

if __libpath__.is_file():
    # Then we open the created shared lib file
    gmfcudalib = ctypes.cdll.LoadLibrary(__libpath__)

    gmfcudalib.gmf.restype = ctypes.c_int
    gmfcudalib.gmf.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_int,
    ]
else:
    raise ImportError(f'{__libpath__} GMF Cuda Library not found')


def gmf_cuda(z_tx, z_rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec, gpu_id=0):
    error_code = gmfcudalib.gmf(
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
        gpu_id,
    )
    assert error_code == 0, f"GMF CUDA-function returned error {error_code}"
