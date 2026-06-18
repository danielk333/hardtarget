"""The Numpy Implementations of the General Matched Filter, or GMF"""

from typing import Any
import numpy as np
import numpy.typing as npt
import scipy.fft as fft
from scipy import optimize

from hardtarget.target_estimation.gmf.types import GMFCfgParams, GMFProParams
from hardtarget.target_estimation.types import MFVariables
from hardtarget.target_estimation.utils import default_mf_vars_items


def dtft_sub_resolution(
    dec_signal: npt.NDArray[np.complexfloating],
    fmin: float,
    fmax: float,
    resolution: int,
) -> npt.NDArray[np.complexfloating]:

    nums = np.arange(len(dec_signal))
    fvec_sub = np.linspace(fmin, fmax, resolution)

    dtft_fractors = np.exp(-1j * 2 * np.pi * fvec_sub[:, None] * nums[None, :])
    dtft = np.abs(np.sum(dtft_fractors * dec_signal[None, :], axis=1)) ** 2
    return dtft


def dtft_solve_with_acceleration(
    decoded_signal: npt.NDArray[np.complexfloating],
    sample_rate: float | int,
    start_freq: float,
    start_accel: float,
    accel_limits: tuple[float, float] = (None, None),
    freq_limits: tuple[float, float] = (None, None),
    method: str = "Nelder-Mead",
    minimize_kwargs: dict[str, Any] | None = {},
) -> float:
    """TODO docstring, this is a bit novel - maybe it works?"""
    t = np.arange(len(decoded_signal)) / sample_rate
    t2 = t**2

    def fun(x):
        dtft_fractors = np.exp(-1j * 2 * np.pi * x[0] * t)
        accel_factors = np.exp(-1j * np.pi * x[1] * t2).astype(np.complex64)
        return -(np.abs(np.mean(dtft_fractors * decoded_signal * accel_factors)) ** 2)

    res = optimize.minimize(
        fun, [start_freq, start_accel], bounds=[freq_limits, accel_limits], method=method, **minimize_kwargs
    )

    # TODO: phase can be computed like this, double check it works and also add it as return value
    dtft_fractors = np.exp(-1j * 2 * np.pi * res.x[0] * t)
    accel_factors = np.exp(-1j * np.pi * res.x[1] * t2).astype(np.complex64)
    phi = np.angle(np.mean(dtft_fractors * decoded_signal * accel_factors))
    return res.x


def dtft_solve(
    decoded_signal: npt.NDArray[np.complexfloating],
    sample_rate: float | int,
    freq_bracket: tuple[float, float],
) -> float:
    """TODO docstring, using the brent method
    NOTE: this is also the baysian MAP - see [^1]

    [^1]: Markkanen, J., Nygren, T., Markkanen, M., 2013.
        New Hight Accuracy Determination of Range and Range Rate of Satellites from EISCAT Radar Data Taken During 2010 SSA CO-VI Campaign,
        in: Proc. 6th European Conference on Space Debris, ESA, Darmstadt, Germany.

    """
    t = np.arange(len(decoded_signal)) / sample_rate

    def fun(x):
        dtft_fractors = np.exp(-1j * 2 * np.pi * x * t)
        return -(np.abs(np.mean(dtft_fractors * decoded_signal)) ** 2)

    res = optimize.minimize_scalar(fun, bracket=freq_bracket, method="brent")

    dtft_fractors = np.exp(-1j * 2 * np.pi * res.x * t)
    phi = np.angle(np.mean(dtft_fractors * decoded_signal))
    return res.x, phi


def _dft_ratio_derivatives_at_zero(d: float) -> tuple[float, float, float, float]:
    """
    #todo docstring, appendix A of [^1]

    Note
    ----
    Eq. A14 in the paper has a duplicated -32/(d*tan(d)^3) term.
    This implementation uses the corrected expression with that term once.

    [^1]: Nygrén, T., Markkanen, J., Aikio, A., Voiculescu, M., 2012.
        High-precision measurement of satellite velocity using the EISCAT radar.
        Ann. Geophys. 30, 1555–1565.
    """
    tan_d = np.tan(d)
    sin_d = np.sin(d)
    cos_d = np.cos(d)

    if np.abs(sin_d) < 1e-15 or np.abs(tan_d) < 1e-15 or np.abs(d) < 1e-15:
        raise ValueError("Ill-conditioned equation; increase FFT length or check pulse length.")

    f1 = 2.0 * (1.0 / d - 1.0 / tan_d)

    f2 = 4.0 * (1.0 / d**2 + 1.0 / tan_d**2 - 2.0 / (d * tan_d))

    f3 = 2.0 * (
        6.0 / d**3
        - 3.0 / d
        - 12.0 / (d**2 * tan_d)
        + 9.0 / (d * tan_d**2)
        + 3.0 / (d * sin_d**2)
        + 2.0 / tan_d
        - 2.0 / tan_d**3
        - 4.0 * cos_d / sin_d**3
    )

    # Correction of A14: the duplicated "-32/(d*tan(d)^3)" is included once.
    f4 = (
        48.0 / d**4
        - 96.0 / (d**3 * tan_d)
        - 24.0 / d**2
        + 72.0 / (d**2 * tan_d**2)
        + 24.0 / (d**2 * sin_d**2)
        + 5.0 / tan_d**4
        - 32.0 / (d * tan_d**3)
        - 10.0 / tan_d**2
        + 32.0 / (d * tan_d)
        + 5.0 / sin_d**4
        - 6.0 / sin_d**2
        + 38.0 / (tan_d**2 * sin_d**2)
        - 64.0 / (d * tan_d * sin_d**2)
        + 1.0
    )

    return f1, f2, f3, f4


def dft_taylor(
    spectrum: npt.NDArray[np.complexfloating],
    signal_len: int,
    sample_rate: float | int,
) -> tuple[float, float]:
    """
    #todo docstring, appendix A of [^1]

    assumes spectrum if shifted

    [^1]: Nygrén, T., Markkanen, J., Aikio, A., Voiculescu, M., 2012.
        High-precision measurement of satellite velocity using the EISCAT radar.
        Ann. Geophys. 30, 1555–1565.
    """

    fft_len = len(spectrum)
    if fft_len == signal_len:
        raise ValueError("FFT length needs to be larger than signal length for expansion to not be singular")

    freqs = fft.fftshift(fft.fftfreq(fft_len, d=1.0))

    m = int(np.argmax(np.abs(spectrum)))
    if m <= 0 or m >= fft_len - 1:
        raise ValueError("FFT peak lies at spectrum edge; cannot use both neighbors.")

    amp_minus = abs(spectrum[m - 1])
    amp_plus = abs(spectrum[m + 1])
    if amp_minus == 0:
        raise ValueError("Left neighbor amplitude is zero; ratio is undefined.")

    R = amp_plus / amp_minus
    nu_m = freqs[m]
    delta_nu = freqs[m + 1] - freqs[m]
    d = np.pi * signal_len * delta_nu

    f1, f2, f3, f4 = _dft_ratio_derivatives_at_zero(d)

    # Eq. A15: (f4/24)x**4 + (f3/6)x**3 + (f2/2)x**2 + f1*x + 1 = R
    coeffs = np.array([f4 / 24.0, f3 / 6.0, f2 / 2.0, f1, 1.0 - R], dtype=float)
    roots = np.roots(coeffs)

    # Physical roots: real root with |x| <= d/2.
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    in_bin = real_roots[np.abs(real_roots) <= abs(d) / 2.0 * (1.0 + 1e-9)]

    if in_bin.size != 1:
        # TODO: figure out if this function should even raise errors or if should just return nan in
        # these cases?
        raise RuntimeError(
            "Expected exactly one real root in [-d/2, d/2]"
            f"([{-d / 2},{d / 2}]), got {in_bin.size}. roots={roots}"
        )

    x = float(in_bin[0])

    nu0 = nu_m + x / (np.pi * signal_len)
    # In Appendix A they changed units from time to samples, change back here
    frequency = nu0 * sample_rate

    Phi_m = np.angle(spectrum[m])
    # Eq. A17: phi = Phi(nu_m) + pi*nu_m*n0 - pi*nu0*(n0+2)
    phase = np.mod(Phi_m + np.pi * nu_m * signal_len - np.pi * nu0 * (signal_len + 2), 2 * np.pi)

    return frequency, phase


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
            # TODO: the rx-tx block size should probably have a padding option? like +-1 for the
            # super resolution stuff, currently not done

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
                # TODO: implement FFT shift in the C and CUDA versions since we need it now?
                ft2 = np.abs(fft.fftshift(fft.fft(dec_signal))) ** 2
                mi = np.argmax(ft2)

                # TODO: we should maybe not return index of doppler frequencies and instead just
                # return the best doppler, that would allow us to do any type of doppler fixing
                if cfg_params.range_rate_sub_resolution > 1:
                    fvec = fft.fftshift(fft.fftfreq(len(dec_signal)))
                    fmin = fvec[mi - 1] if mi >= 1 else fvec[0]
                    fmax = fvec[mi + 1] if mi < len(fvec) - 1 else fvec[-1]
                    dtft = dtft_sub_resolution(dec_signal, fmin, fmax, cfg_params.range_rate_sub_resolution)
                    dtft_ind = np.argmax(dtft)
                    ftmax = dtft[dtft_ind]
                    ftind = mi * cfg_params.range_rate_sub_resolution + dtft_ind
                else:
                    ftmax = ft2[mi]
                    ftind = mi

                if ftmax > vals[index]:
                    vals[index] = ftmax
                    # index of doppler that gives highest integrated energy at this range gate
                    v_ind[index] = ftind
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
