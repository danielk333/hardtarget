"""The Numpy Implementation of the Discrete Polynomial-phase Transform, or DPT"""

import numpy as np
import numpy.typing as npt
import scipy.fft as fft

from hardtarget.matched_filter.dpt.types import DPTCfgParams, DPTProParams
from hardtarget.matched_filter.types import MFVariables
from hardtarget.matched_filter.utils import default_mf_vars_items


def fast_dpt_np(
    z_tx: npt.NDArray[np.complexfloating],
    z_rx: npt.NDArray[np.complexfloating],
    tx_pwr: npt.NDArray,
    cfg_params: DPTCfgParams,
    pro_params: DPTProParams,
) -> MFVariables:
    """
    Development version of GMF using discrete ambiguity function spectrum to speed up acceleration search

    Args:
        z_tx: Transmitted signal
        z_rx: Recived signal
        tx_pwr: Power of the transmission
        cfg_params: DPT configuration parameters
        pro_params: DPT processing parameters

    Returns
        MFVariables from the analysis
    """

    size = (len(pro_params.ranges),)
    dc, vals, v_ind, a_ind = default_mf_vars_items(size)

    for ri, rg in enumerate(pro_params.rel_rgs):
        for sub_res in range(cfg_params.range_gate_sub_resolution):
            drg = int(rg // cfg_params.frequency_decimation)
            zr = z_rx[pro_params.il1_rx_window_indices + rg]
            # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
            echo = np.sum((zr * z_tx[:, sub_res]).reshape(-1, cfg_params.frequency_decimation), axis=-1)
            dec_signal = np.zeros((pro_params.decimated_read_length,), dtype=np.complex64)
            dec_signal[pro_params.il0_dec_rx_window_indices + drg] = echo
            index = sub_res + ri * cfg_params.range_gate_sub_resolution
            dc[index] = np.abs(np.sum(echo)) ** 2

            dpt2 = dec_signal[pro_params.decimated_ipp_delay_parameter :] * np.conj(
                dec_signal[: -pro_params.decimated_ipp_delay_parameter]
            )
            dpt2_spec = np.abs(fft.fftshift(fft.fft(dpt2)))
            dspec_peak = np.argmax(dpt2_spec)

            dec_signal[pro_params.il0_dec_rx_window_indices + drg] = (
                pro_params.acceleration_phasors[dspec_peak] * echo
            )
            ft2 = np.abs(fft.fft(dec_signal)) ** 2
            spec_peak = np.argmax(ft2)

            vals[index] = ft2[spec_peak]
            v_ind[index] = spec_peak
            a_ind[index] = dspec_peak

    return MFVariables(vals=vals, dc=dc, v_ind=v_ind, a_ind=a_ind, tx_pwr=tx_pwr)
