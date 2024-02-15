import ctypes
import sysconfig
import pathlib
import logging

from .numpy_ctypes import nptype

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
        nptype("c8", 1), ctypes.c_int,  # 1, 2
        nptype("c8", 1), ctypes.c_int,  # 3, 4
        nptype("c8", 2), ctypes.c_int,  # 5, 6
        nptype("i4", 1), ctypes.c_int,  # 7, 8
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
        z_tx,  # 1
        z_tx.size,  # 2
        z_rx,  # 3
        z_rx.size,  # 4
        acc_phasors,  # 5
        acc_phasors.shape[0],  # 6
        rgs,  # 7
        rgs.size,  # 8
        frequency_decimation,  # 9
        gmf_variables.vals,  # 10
        gmf_variables.dc,  # 11
        gmf_variables.v_ind,  # 12
        gmf_variables.a_ind,  # 13
        rx_window_indices,  # 14
        dec_rx_window_indices,  # 15
        dec_signal_len,  # 16
        gpu_id,  # 17
    )
    assert error_code == 0, f"GMF CUDA-function returned error {error_code}"
