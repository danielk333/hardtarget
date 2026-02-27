"""The Numpy Implementations of the General Matched Filter, or GMF"""

import numpy as np
import numpy.typing as npt
import scipy.fft as fft

from hardtarget.matched_filter.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.matched_filter.types import MFVariables
from hardtarget.matched_filter.utils import default_mf_vars_items


def fast_gmf_np(
    tx: npt.NDArray[np.complexfloating],
    rx: npt.NDArray[np.complexfloating],
    tx_pwr: npt.NDArray[np.floating],
    cfg_params: GMFCfgParams,
    pro_params: GMFProParams,
) -> MFVariables:
    """
    Compute the output of the Generalized Matched Filter GMF

    Args:
        tx: Transmitted signal
        rx: Recived signal
        tx_pwr: Power of the transmission
        cfg_params: DPT configuration parameters
        pro_params: DPT processing parameters

    Returns
        MFVariables from the analysis
    """
    # number of range gates is input from user
    n_acc = pro_params.fgmf_acceleration_phasors.shape[0]

    size = (len(pro_params.ranges),)
    dc, vals, v_ind, a_ind = default_mf_vars_items(size)

    for ri, rg in enumerate(pro_params.rel_rgs):
        for sub_res in range(cfg_params.range_gate_sub_resolution):
            drg = int(rg // cfg_params.frequency_decimation)
            zr = rx[pro_params.il1_rx_window_indices + rg]
            # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
            echo = np.sum((zr * tx[:, sub_res]).reshape(-1, cfg_params.frequency_decimation), axis=-1)
            dec_signal = np.zeros((pro_params.decimated_read_length,), dtype=np.complex64)
            # zero-frequency (DC) is used to get range-dependent noise floor
            index = sub_res + ri * cfg_params.range_gate_sub_resolution
            dc[index] = np.abs(np.sum(echo)) ** 2

            for ai in range(n_acc):
                dec_signal[pro_params.il0_dec_rx_window_indices + drg] = (
                    pro_params.fgmf_acceleration_phasors[ai] * echo
                )
                ft2 = np.abs(fft.fft(dec_signal)) ** 2
                mi = np.argmax(ft2)

                if ft2[mi] > vals[index]:
                    vals[index] = ft2[mi]
                    # index of doppler that gives highest integrated energy at this range gate
                    v_ind[index] = mi
                    # index of acceleration that gives highest integrated energy at this range gate
                    a_ind[index] = pro_params.inds_accelerations[ai]

    return MFVariables(vals=vals, dc=dc, v_ind=v_ind, a_ind=a_ind, tx_pwr=tx_pwr)


def fast_gmf_no_reduce_np(
    tx: npt.NDArray[np.complexfloating],
    rx: npt.NDArray[np.complexfloating],
    tx_pwr: npt.NDArray,
    cfg_params: GMFCfgParams,
    pro_params: GMFProParams,
) -> MFVariables:
    """Slow development version of gmf to see otherwise reduced dimensions

    WARNING: Not working

    Args:
        tx: Transmitted signal
        rx: Recived signal
        tx_pwr: Power of the transmission
        cfg_params: DPT configuration parameters
        pro_params: DPT processing parameters

    Returns
        MFVariables from the analysis
    """

    # TODO: Correct this function, looks to not have worked in a while

    return MFVariables(
        vals=np.empty((1), dtype=np.float32),
        dc=np.empty((1), dtype=np.float32),
        v_ind=np.empty((1), dtype=np.int32),
        a_ind=np.empty((1), dtype=np.int32),
        tx_pwr=np.empty((1), dtype=np.float32),
    )


"""
    ra = params.pro.reduce_axis

    size = (params.pro.n_ranges,)
    dc, vals, v_ind, a_ind = default_mf_vars_items(size)
    r_ind = []

    # number of range gates is input from user
    n_acc = params.der.acceleration_phasors.shape[0]
    for ri, rg in enumerate(params.der.rgs):
        zr = rx[params.der.rx_window_indices + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * tx).reshape(-1, params.pro.frequency_decimation), axis=-1)
        decimated_signal = np.zeros(
            (params.der.dec_signal_length,), dtype=np.complex64
        )

        for ai in range(n_acc):
            decimated_signal[params.der.dec_rx_window_indices] = params.der.acceleration_phasors[ai] * echo
            _gmfo = np.abs(fft.fft(decimated_signal)) ** 2
            if ai == 0:
                # gmf_dc_vec is the range-dependent noise floor
                dc[ri] = _gmfo[0]
            if ra[1]:
                vi = np.argmax(_gmfo)
                new_val = _gmfo[vi]
                if ra[0]:
                    if new_val > vals[ai]:
                        vals[ai] = new_val
                        r_ind[ai] = ri
                        v_ind[ai] = vi
                elif ra[2]:
                    if new_val > vals[ri]:
                        vals[ri] = new_val
                        v_ind[ri] = vi
                        a_ind[ri] = ai
                else:
                    raise NotImplementedError("")
            else:
                inds = np.arange(len(_gmfo))
                if ra[0] and ra[2]:
                    vals_stack = np.stack([vals[:], _gmfo[:]])
                    mi = np.argmax(vals_stack, axis=0)
                    vals[:] = vals_stack[mi, inds]
                    sel = mi == 1
                    r_ind[sel] = ri
                    a_ind[sel] = ai
                elif ra[0]:
                    vals_stack = np.stack([vals[:, ai], _gmfo[:]])
                    mi = np.argmax(vals_stack, axis=0)
                    vals[:, ai] = vals_stack[mi, inds]
                    sel = mi == 1
                    r_ind[sel, ai] = ri
                else:
                    vals_stack = np.stack([vals[:, ri], _gmfo[:]])
                    mi = np.argmax(vals_stack, axis=0)
                    vals[:, ri] = vals_stack[mi, inds]
                    sel = mi == 1
                    a_ind[sel, ri] = ai

    return MFVariables(dc=dc, vals=vals, v_ind=v_ind, a_ind=a_ind, tx_pwr=tx_pwr)
"""
