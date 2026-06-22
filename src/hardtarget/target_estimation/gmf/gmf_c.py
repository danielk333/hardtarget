"""Wrapper for the C implementation of the General Matched Filter, or GMF"""

import numpy as np
import numpy.typing as npt

from hardtarget.libs import load_c_lib
from hardtarget.target_estimation.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.target_estimation.types import MFVariables
from hardtarget.target_estimation.utils import default_mf_vars_items

mfclib = load_c_lib()


def fast_gmf_c(
    tx: npt.NDArray[np.complex64],
    rx: npt.NDArray[np.complex64],
    tx_pwr: npt.NDArray,
    cfg_params: GMFCfgParams,
    pro_params: GMFProParams,
) -> MFVariables:
    """
    Wrapper for the Fast GMF C implementation

    Args:
        tx: Transmitted signal
        rx: Recived signal
        tx_pwr: Power of the transmission
        cfg_params: GMF configuration parameters
        pro_params: GMF processing parameters

    Returns
        MFVariables from the analysis
    """

    size = (len(pro_params.ranges),)
    dc, vals, v_ind, a_ind, phi = default_mf_vars_items(size)

    tx = tx.flatten()
    # Below comments are for ctypes argument error debugging
    error_code = mfclib.fgmf(
        tx,  # 1
        int(tx.size / cfg_params.range_gate_sub_resolution),  # 2
        rx,  # 3
        rx.size,  # 4
        cfg_params.range_gate_sub_resolution,  # 5
        pro_params.fgmf_acceleration_phasors,  # 6
        pro_params.fgmf_acceleration_phasors.shape[0],  # 7
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
    )
    a_ind[:] = pro_params.inds_accelerations[a_ind]
    assert error_code == 0, f"GMF C-function returned error {error_code}"

    return MFVariables(vals=vals, dc=dc, v=v_ind, a=a_ind, phi=phi, tx_pwr=tx_pwr)
