"""Wrapper for the Cuda implementation of the General Matched Filter, or GMF

!!! Warning
    Not working, has not yet been refactored to support sub resolution.

"""

import numpy as np
import numpy.typing as npt
from radardef.types import ExpDef

from hardtarget.libs import load_cuda_lib
from hardtarget.target_estimation.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.target_estimation.types import MFVariables
from hardtarget.target_estimation.utils import default_mf_vars_items

gmfcudalib = load_cuda_lib()


def print_cuda_devices() -> None:
    """Print available cuda devices"""
    gmfcudalib.print_devices()


def test_cuda() -> None:
    """Tests if cuda works"""
    assert gmfcudalib.test_cuda() == 0, "CUDA DIDNT WORK"


def fast_gmf_cuda(
    tx: npt.NDArray[np.complexfloating],
    rx: npt.NDArray[np.complexfloating],
    tx_pwr: npt.NDArray,
    exp_def: ExpDef,
    cfg_params: GMFCfgParams,
    pro_params: GMFProParams,
    gpu_id: int = 0,
) -> MFVariables:
    """
    Wrapper for the Fast GMF Cuda implementation

    Args:
        tx: Transmitted signal
        rx: Recived signal
        tx_pwr: Power of the transmission
        cfg_params: DPT configuration parameters
        pro_params: DPT processing parameters
        gpu_id: GPU identification

    Returns
        MFVariables from the analysis
    """

    size = (len(pro_params.ranges),)
    dc, vals, v_ind, a_ind, phi = default_mf_vars_items(size)

    error_code = gmfcudalib.gmf(
        tx,  # 1
        tx.size,  # 2
        rx,  # 3
        rx.size,  # 4
        pro_params.fgmf_acceleration_phasors,  # 5
        pro_params.fgmf_acceleration_phasors.shape[0],  # 6
        pro_params.rel_rgs,  # 7
        pro_params.rel_rgs.size,  # 8
        cfg_params.frequency_decimation,  # 9
        vals,  # 10
        dc,  # 11
        v_ind,  # 12
        a_ind,  # 13
        pro_params.il1_rx_window_indices,  # 14
        pro_params.il0_dec_rx_window_indices.astype("i4"),  # 15
        pro_params.decimated_read_length,  # 16
        gpu_id,  # 17
    )
    a_ind[:] = pro_params.inds_accelerations[a_ind]

    if error_code != 0:
        raise Exception(f"GMF CUDA-function returned error {error_code}")

    return MFVariables(vals=vals, dc=dc, v=v_ind, a=a_ind, phi=phi, tx_pwr=tx_pwr)
