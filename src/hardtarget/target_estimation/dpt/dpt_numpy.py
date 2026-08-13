"""The Numpy Implementation of the Discrete Polynomial-phase Transform, or DPT"""

import numpy as np
import numpy.typing as npt
import scipy.fft as fft
from radardef.types import ExpDef

from hardtarget.target_estimation.dpt.types import DPTCfgParams, DPTProParams
from hardtarget.target_estimation.dtft_solvers import dtft_solve, dtft_solve_with_acceleration
from hardtarget.target_estimation.types import MFVariables
from hardtarget.target_estimation.utils import default_mf_vars_items


def fast_dpt_np(
    tx: npt.NDArray[np.complexfloating],
    rx: npt.NDArray[np.complexfloating],
    tx_pwr: npt.NDArray[np.floating],
    exp_def: ExpDef,
    cfg_params: DPTCfgParams,
    pro_params: DPTProParams,
) -> MFVariables:
    """
    Development version of GMF using discrete ambiguity function spectrum to speed up acceleration search

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
    dc, vals, v, a, phi = default_mf_vars_items(size)

    for ri, rg in enumerate(pro_params.rel_rgs):
        for sub_res in range(cfg_params.range_gate_sub_resolution):
            drg = int(rg // cfg_params.frequency_decimation)
            zr = rx[pro_params.il1_rx_window_indices + rg]
            # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
            echo = np.sum((zr * tx[:, sub_res]).reshape(-1, cfg_params.frequency_decimation), axis=-1)
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

            ft = fft.fftshift(fft.fft(dec_signal))
            ft2 = np.abs(ft) ** 2
            spec_peak = np.argmax(ft2)

            vals[index] = ft2[spec_peak]
            v[index] = pro_params.fft_frequencies[spec_peak]
            a[index] = pro_params.accelerations[dspec_peak]
            phi[index] = np.angle(ft[spec_peak])

            # Refine acceleration and phase
            v_index_p = spec_peak
            if spec_peak < len(pro_params.fft_frequencies) - 1:
                v_index_p += 1
            v_index_m = spec_peak
            if spec_peak > 0:
                v_index_m = spec_peak - 1
            if cfg_params.refine_acceleration:
                pwr, f_est, a_est, phi_est = dtft_solve_with_acceleration(
                    decoded_signal=dec_signal,
                    sample_rate=exp_def.sample_rate,
                    start_freq=pro_params.fft_frequencies[spec_peak],
                    # TODO: i think this accel is the wrong variable and the wrong units, verify
                    start_accel=pro_params.accelerations[dspec_peak],
                    freq_limits=(
                        pro_params.fft_frequencies[v_index_m],
                        pro_params.fft_frequencies[v_index_p],
                    ),
                    accel_limits=(
                        pro_params.accelerations[dspec_peak - 1],
                        pro_params.accelerations[dspec_peak + 1],
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

    # TODO: add phase here and fix return values to not be inds, use dtft opt to get v and phase
    return MFVariables(
        vals=vals,
        dc=dc,
        v=v,
        a=a,
        phi=phi,
        tx_pwr=tx_pwr,
    )
