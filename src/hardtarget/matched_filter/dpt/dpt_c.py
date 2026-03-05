"""Wrapper for the C implementation of the Discrete Polynomial-phase Transform, or DPT"""

import numpy as np
import numpy.typing as npt

from hardtarget.matched_filter.dpt.types import DPTCfgParams, DPTProParams
from hardtarget.matched_filter.types import MFVariables
from hardtarget.matched_filter.utils import default_mf_vars_items, load_c_lib

mfclib = load_c_lib()


def fast_dpt_c(
    tx: npt.NDArray[np.complex64],
    rx: npt.NDArray[np.complex64],
    tx_pwr: npt.NDArray,
    cfg_params: DPTCfgParams,
    pro_params: DPTProParams,
) -> MFVariables:
    """
    Wrapper for the Fast DPT C implementation

    Args:
        tx: Transmitted signal
        rx: Recived signal
        tx_pwr: Power of the transmission
        cfg_params: DPT configuration parameters
        pro_params: DPT processing parameters

    Returns
        MFVariables from the analysis
    """

    size = (len(pro_params.ranges),)
    dc, vals, v_ind, a_ind = default_mf_vars_items(size)

    tx = tx.flatten()
    # Below comments are for ctypes argument error debugging
    error_code = mfclib.fdpt(
        tx,  # 1
        int(tx.size / cfg_params.range_gate_sub_resolution),  # 2
        rx,  # 2
        rx.size,  # 4
        cfg_params.range_gate_sub_resolution,  # 5
        pro_params.acceleration_phasors,  # 6
        pro_params.acceleration_phasors.shape[0],  # 7
        pro_params.rel_rgs,  # 8
        pro_params.rel_rgs.size,  # 9
        cfg_params.frequency_decimation,  # 10
        vals,  # 11
        dc,  # 12
        v_ind,  # 13
        a_ind,  # 14
        pro_params.il1_rx_window_indices,  # 15
        pro_params.il0_dec_rx_window_indices.astype("i4"),  # 16
        pro_params.decimated_read_length,  # 17
        pro_params.decimated_ipp_delay_parameter,  # 18
    )

    if error_code != 0:
        raise Exception(f"DPT C-function returned error {error_code}")

    return MFVariables(vals=vals, dc=dc, v_ind=v_ind, a_ind=a_ind, tx_pwr=tx_pwr)
