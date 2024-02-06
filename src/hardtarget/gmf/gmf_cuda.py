import ctypes
import sysconfig
import pathlib
import logging

logger = logging.getLogger(__name__)

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
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]

    gmfcudalib.print_devices.restype = None
    gmfcudalib.print_devices.argtypes = []
else:
    raise ImportError(f'{__libpath__} GMF Cuda Library not found')


def print_cuda_devices():
    """Print available cuda devices
    """
    gmfcudalib.print_devices()


def gmfcu(z_tx, z_rx, gmf_variables, gmf_params, gpu_id=0):
    acc_phasors = gmf_params["DER"]["acceleration_phasors"]
    rgs = gmf_params["DER"]["rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    rx_window_indices = gmf_params["DER"]["rx_window_indices"]
    dec_rx_window_indices = gmf_params["DER"]["dec_rx_window_indices"]
    dec_signal_len = gmf_params["DER"]["dec_signal_length"]

    error_code = gmfcudalib.gmf(
        z_tx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        z_tx.size,
        z_rx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        z_rx.size,
        acc_phasors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acc_phasors.shape[0],
        rgs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        rgs.size,
        frequency_decimation,
        gmf_variables.vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gmf_variables.dc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gmf_variables.v_ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        gmf_variables.a_ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        rx_window_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dec_rx_window_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dec_signal_len,
        gpu_id,
    )
    assert error_code == 0, f"GMF CUDA-function returned error {error_code}"
