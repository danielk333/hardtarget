from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.fft as fft
from scipy import optimize


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
    accel_limits: tuple[float | None, float | None] = (None, None),
    freq_limits: tuple[float | None, float | None] = (None, None),
    method: str = "Nelder-Mead",
    minimize_kwargs: dict[str, Any] = {},
) -> tuple[float, float, float, float]:
    """TODO docstring, this is a bit novel - maybe it works?"""
    t = np.arange(len(decoded_signal)) / sample_rate
    t2 = t**2

    def fun(x: tuple[float, float]) -> float:
        """
        Args:
            x: frequence, acceleration
        """

        dtft_fractors = np.exp(-1j * 2 * np.pi * x[0] * t)
        accel_factors = np.exp(-1j * np.pi * x[1] * t2).astype(np.complex64)
        return -(np.abs(np.sum(dtft_fractors * decoded_signal * accel_factors)) ** 2)

    res = optimize.minimize(
        fun,
        [start_freq, start_accel],
        bounds=[freq_limits, accel_limits],
        method=method,
        **minimize_kwargs,
    )

    dtft_fractors = np.exp(-1j * 2 * np.pi * res.x[0] * t)
    accel_factors = np.exp(-1j * np.pi * res.x[1] * t2).astype(np.complex64)
    phi = np.angle(np.sum(dtft_fractors * decoded_signal * accel_factors))

    return -res.fun, res.x[0], res.x[1], phi


def dtft_solve(
    decoded_signal: npt.NDArray[np.complexfloating],
    sample_rate: float | int,
    freq_bracket: tuple[float, float],
) -> tuple[float, float, float]:
    """TODO docstring, using the brent method
    NOTE: this is also the baysian MAP - see [^1]

    [^1]: Markkanen, J., Nygren, T., Markkanen, M., 2013.
        New Hight Accuracy Determination of Range and Range Rate of Satellites from EISCAT Radar Data Taken During 2010 SSA CO-VI Campaign,
        in: Proc. 6th European Conference on Space Debris, ESA, Darmstadt, Germany.

    """
    t = np.arange(len(decoded_signal)) / sample_rate

    def fun(x: float) -> float:

        dtft_fractors = np.exp(-1j * 2 * np.pi * x * t)
        return -(np.abs(np.sum(dtft_fractors * decoded_signal)) ** 2)

    res = optimize.minimize_scalar(fun, bracket=freq_bracket, method="brent")

    dtft_fractors = np.exp(-1j * 2 * np.pi * res.x * t)
    phi = np.angle(np.sum(dtft_fractors * decoded_signal))
    return -res.fun, res.x, phi


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

    assumes spectrum is shifted

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
