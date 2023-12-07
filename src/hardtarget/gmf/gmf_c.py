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

    # TODO: we should use numpy ctypes here for arrays, that would
    # error if we accidentally pass the wrong dtype
    # the current `data_as` usage will reinterpret cast which is unsafe
    gmfclib.gmf.restype = ctypes.c_int
    gmfclib.gmf.argtypes = [
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
    ]
else:
    raise ImportError(f'{__libpath__} GMF C Library not found')


def gmfc(z_tx, z_rx, gmf_variables, gmf_params):
    
    acc_phasors = gmf_params["DER"]["acceleration_phasors"]
    rgs = gmf_params["DER"]["rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    rx_window_indices = gmf_params["DER"]["rx_window_indices"]

    # TODO: generalize the preprocess filtering of 0 tx power
    # since it can cause unnessary slowdowns depending on experiment setup

    error_code = gmfclib.gmf(
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
    )
    assert error_code == 0, f"GMF C-function returned error {error_code}"
