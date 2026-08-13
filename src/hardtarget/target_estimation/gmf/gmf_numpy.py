"""The Numpy Implementations of the General Matched Filter, or GMF"""

import numpy as np
import numpy.typing as npt
import scipy.fft as fft
from radardef.types import ExpDef

from hardtarget.target_estimation.dtft_solvers import dtft_solve, dtft_solve_with_acceleration
from hardtarget.target_estimation.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.target_estimation.types import MFVariables
from hardtarget.target_estimation.utils import default_mf_vars_items


def fast_gmf_np(
    tx: npt.NDArray[np.complexfloating],
    rx: npt.NDArray[np.complexfloating],
    tx_pwr: npt.NDArray[np.floating],
    exp_def: ExpDef,
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
    dc, vals, v, a, phi = default_mf_vars_items(size)

    for ri, rg in enumerate(pro_params.rel_rgs):
        for sub_res in range(cfg_params.range_gate_sub_resolution):
            drg = int(rg // cfg_params.frequency_decimation)
            zr = rx[pro_params.il1_rx_window_indices + rg]
            # TODO: the rx-tx block size should probably have a padding option? like +-1 for the
            # super resolution stuff, currently not done, needs more investigation if needed

            # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
            echo = np.sum((zr * tx[:, sub_res]).reshape(-1, cfg_params.frequency_decimation), axis=-1)
            dec_signal = np.zeros((pro_params.decimated_read_length,), dtype=np.complex64)
            # zero-frequency (DC) is used to get range-dependent noise floor
            index = sub_res + ri * cfg_params.range_gate_sub_resolution
            dc[index] = np.abs(np.sum(echo)) ** 2
            a_index = -1
            v_index = -1

            for ai in range(n_acc):
                dec_signal[pro_params.il0_dec_rx_window_indices + drg] = (
                    pro_params.fgmf_acceleration_phasors[ai] * echo
                )
                ft = fft.fftshift(fft.fft(dec_signal))
                ft2 = np.abs(ft) ** 2
                mi = np.argmax(ft2)
                pwr, f_est, phi_est = ft2[mi], pro_params.fft_frequencies[mi], np.angle(ft[mi])

                if pwr > vals[index]:
                    vals[index] = pwr
                    # TODO: the convention should be that these are signal specific numbers, i.e.
                    # doppler frequency and cycle acceleration (i.e. cycle/s and cycle/s^2), not
                    # physical units like m/s och m/s^2

                    # doppler that gives highest integrated energy at this range gate
                    v[index] = f_est
                    v_index = int(mi)
                    # acceleration that gives highest integrated energy at this range gate
                    a[index] = pro_params.accelerations[ai]
                    a_index = ai
                    # phase at the best acceleration and range rate
                    phi[index] = phi_est

            # Refine acceleration and phase
            dec_signal[pro_params.il0_dec_rx_window_indices + drg] = (
                pro_params.fgmf_acceleration_phasors[a_index] * echo
            )
            v_index_p = v_index
            if v_index < len(pro_params.fft_frequencies) - 1:
                v_index_p += 1
            v_index_m = v_index
            if v_index > 0:
                v_index_m = v_index - 1
            if cfg_params.refine_acceleration:
                pwr, f_est, a_est, phi_est = dtft_solve_with_acceleration(
                    decoded_signal=dec_signal,
                    sample_rate=exp_def.sample_rate,
                    start_freq=pro_params.fft_frequencies[v_index],
                    # TODO: i think this accel is the wrong variable and the wrong units, verify
                    start_accel=pro_params.accelerations[a_index],
                    freq_limits=(
                        pro_params.fft_frequencies[v_index_m],
                        pro_params.fft_frequencies[v_index_p],
                    ),
                    accel_limits=(
                        pro_params.accelerations[a_index - 1],
                        pro_params.accelerations[a_index + 1],
                    ),
                )
                v[index] = f_est
                a[index] = a_est
                phi[index] = phi_est
            elif cfg_params.refine_doppler:
                pwr, f_est, phi_est = dtft_solve(
                    decoded_signal=dec_signal,
                    sample_rate=exp_def.sample_rate,
                    freq_bracket=(
                        pro_params.fft_frequencies[v_index_m],
                        pro_params.fft_frequencies[v_index_p],
                    ),
                )
                v[index] = f_est
                phi[index] = phi_est

    return MFVariables(
        vals=vals,
        dc=dc,
        v=v,
        a=a,
        tx_pwr=tx_pwr,
        phi=phi,
    )


def fast_gmf_no_reduce_np(
    tx: npt.NDArray[np.complexfloating],
    rx: npt.NDArray[np.complexfloating],
    tx_pwr: npt.NDArray,
    exp_def: ExpDef,
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
        vals=np.empty((1), dtype=np.float64),
        dc=np.empty((1), dtype=np.float64),
        v=np.empty((1), dtype=np.float64),
        a=np.empty((1), dtype=np.float64),
        phi=np.empty((1), dtype=np.float64),
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
