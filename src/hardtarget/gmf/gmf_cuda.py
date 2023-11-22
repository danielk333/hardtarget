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
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
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
    # TODO: gpu_id should be option somehow or should be set by mpi
    # it depends on the node / gpu-count layout of the cluster
    acc_phasors = gmf_params["acceleration_phasors"]
    rgs = gmf_params["rgs"]
    frequency_decimation = gmf_params["frequency_decimation"]
    rx_window_indices = gmf_params["rx_window_indices"]

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
        gpu_id,
    )
    assert error_code == 0, f"GMF CUDA-function returned error {error_code}"
