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
__libpath__ = pymodule_dir / ('gmfclib' + suffix)

if __libpath__.is_file():
    # Then we open the created shared lib file
    gmfclib = ctypes.cdll.LoadLibrary(__libpath__)

    gmfclib.gmf.restype = ctypes.c_int
    # TODO: rename these to better names
    # - probably gmf -> fast-gmf
    # full amplitude domain function -> gmf
    # discrete polynomial phase transform -> fast-dpt
    gmfclib.gmf.argtypes = [
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
    ]
else:
    raise ImportError(f'{__libpath__} GMF C Library not found')


def fast_gmf_c(z_tx, z_rx, gmf_variables, gmf_params):
    acc_phasors = gmf_params["DER"]["fgmf_acceleration_phasors"]
    acc_inds = gmf_params["DER"]["inds_accelerations"]
    rel_rgs = gmf_params["DER"]["rel_rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    il1_rx_win = gmf_params["DER"]["il1_rx_window_indices"]
    il0_dec_rx_win = gmf_params["DER"]["il0_dec_rx_window_indices"].astype("i4")
    dec_signal_len = gmf_params["PRO"]["decimated_read_length"]

    # Below comments are for ctypes argument error debugging
    error_code = gmfclib.gmf(
        z_tx,  # 1
        z_tx.size,  # 2
        z_rx,  # 3
        z_rx.size,  # 4
        acc_phasors,  # 5
        acc_phasors.shape[0],  # 6
        rel_rgs,  # 7
        rel_rgs.size,  # 8
        frequency_decimation,  # 9
        gmf_variables.vals,  # 10
        gmf_variables.dc,  # 11
        gmf_variables.v_ind,  # 12
        gmf_variables.a_ind,  # 13
        il1_rx_win,  # 14
        il0_dec_rx_win,  # 15
        dec_signal_len,  # 16
    )
    gmf_variables.a_ind[:] = acc_inds[gmf_variables.a_ind]
    assert error_code == 0, f"GMF C-function returned error {error_code}"
