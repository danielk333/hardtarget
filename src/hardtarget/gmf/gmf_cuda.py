import ctypes as C
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
    gmfcudalib = C.cdll.LoadLibrary(__libpath__)

    gmfcudalib.gmf.restype = C.c_int
    gmfcudalib.gmf.argtypes = [
        C.POINTER(C.c_float), C.c_int, 
        C.POINTER(C.c_float), C.c_int, 
        C.POINTER(C.c_float), C.c_int, 
        C.POINTER(C.c_float), C.c_int, C.c_int, 
        C.POINTER(C.c_float), C.POINTER(C.c_float), 
        C.POINTER(C.c_float), C.POINTER(C.c_float), 
        C.c_int,
    ]
else:
    raise ImportError(f'{__libpath__} GMF Cuda Library not found')


def gmf_cuda(z_tx, z_rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec, rank=0):
    txlen=int(len(z_tx))
    rxlen=len(z_rx)
    a=gmfcudalib.gmf(
        z_tx.ctypes.data_as(C.POINTER(C.c_float)),
        txlen,
        z_rx.ctypes.data_as(C.POINTER(C.c_float)),
        rxlen,
        acc_phasors.ctypes.data_as(C.POINTER(C.c_float)),
        int(acc_phasors.shape[0]),
        rgs.ctypes.data_as(C.POINTER(C.c_float)),
        len(rgs),
        int(dec),
        gmf_vec.ctypes.data_as(C.POINTER(C.c_float)),
        gmf_dc_vec.ctypes.data_as(C.POINTER(C.c_float)),
        v_vec.ctypes.data_as(C.POINTER(C.c_float)),
        a_vec.ctypes.data_as(C.POINTER(C.c_float)),
        rank,
    )