"""
Tx signal models, used to simulate tx signals when not available.
"""

import numpy as np
import numpy.typing as npt
import scipy.interpolate as interpolate
import scipy.signal as sc_signal
from scipy.fft import fft, fftfreq, ifft

from hardtarget.constants import FIRFilter


def boxcar(n: int, normalize: bool = True) -> npt.NDArray:
    h = np.ones(n, dtype=float)
    return h / h.sum() if normalize else h


def apply_b414d15_gaus(x: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
    """
    Equivalent chain from b414d15_gaus.fir:
      total decimation = 15

      5 cascaded 5-tap boxcar FIRs
      decimate by 5
      2 cascaded 2-tap boxcar FIRs
      decimate by 3
    """

    y = np.asarray(x)

    # HDF section: 5 boxcar FIRs, each 5 taps
    h5 = boxcar(5)
    for _ in range(5):
        y = sc_signal.lfilter(h5, [1.0], y)  # type: ignore[attr-defined]

    # Decimate x MHz -> x/5 MHz
    y = y[::5]

    # FIR section: 2 boxcar FIRs, each 2 taps
    h2 = boxcar(2)
    for _ in range(2):
        y = sc_signal.lfilter(h2, [1.0], y)  # type: ignore[attr-defined]

    # Decimate x/5 MHz -> x/15 MHz
    y = y[::3]

    return y


def cic_decimate(
    x: npt.NDArray[np.complexfloating], decimation: int, combs: int, delay: int = 1
) -> npt.NDArray[np.complexfloating]:
    """
    CIC decimator: N integrators at input rate that is decimated,
    then N comb stages at output rate with a delay.
    """
    y = np.asarray(x, dtype=np.complex128)

    # integrators
    for _ in range(combs):
        y = np.cumsum(y)

    # decimate
    y = y[::decimation]

    # combs
    for _ in range(combs):
        y = y - np.concatenate([np.zeros(delay, dtype=y.dtype), y[:-delay]])

    # normalize CIC DC gain
    y /= (decimation * delay) ** combs
    return y


def mu_radar_filter_post_2004(
    x: npt.NDArray[np.complexfloating],
    t_samp_usec: float = 6.0,
) -> npt.NDArray[np.complexfloating]:
    """
    Model MUR chain accoring to [^1]
    IF samples at complex baseband -> CIC decimation -> 16-tap FIR compensation.

    [^1]: Hassenpflug, G., Yamamoto, M., Luce, H., Fukao, S., 2008.
        Description and demonstration of the new Middle and Upper atmosphere Radar imaging system: 1-D, 2-D, and 3-D imaging of troposphere and stratosphere.
        Radio Sci. 43, RS2013. https://doi.org/10.1029/2006RS003603

    """
    # CIC matched-filter / decimator
    # TODO: guessing the cic decimation rate of 8, it kinda makes sense beacuse with 16 taps
    # 8*15=120 which is the total decimation rate... but double check needed!
    y_cic = cic_decimate(x, decimation=8, combs=6)

    # 16-tap FIR amplitude/frequency compensator
    # TODO: gussing the compensating FIR, no coefficients were available in the paper?
    fir_taps = sc_signal.firwin(  # type: ignore[attr-defined]
        numtaps=16,
        cutoff=0.8,
        window="hamming",
    )
    y_out = sc_signal.lfilter(fir_taps, [1.0], y_cic)  # type: ignore[attr-defined]
    y_out = y_out[::15]

    return y_out


def match_pulse_code(
    tx_signal: npt.NDArray[np.complex128],
    codes: npt.NDArray[np.float64],
    baud_length_usec: int,
    t_samp_usec: int,
    ipp_t_usec: int,
) -> npt.NDArray[np.float64]:

    matches = np.zeros((codes.shape[0],), dtype=np.float64)
    for ind in range(len(matches)):
        signal = simulate_pulse_code(
            code=codes[ind, :],
            baud_length_usec=baud_length_usec,
            t_samp_usec=t_samp_usec,
            ipp_t_usec=ipp_t_usec,
            signal_length=len(tx_signal),
        )
        matches[ind] = np.abs(np.sum(tx_signal * np.conj(signal)))
    return matches


def tx_modulation_model(
    tx_signal: npt.NDArray[np.complex128],
    tx_stencil: npt.NDArray[np.bool],
    fir_filter: FIRFilter = FIRFilter.b414d15_gaus,
    sub_resolution: int = 1,
    kind: str = "linear",
) -> npt.NDArray[np.complex128]:
    modulated_tx = np.zeros((tx_signal.size, sub_resolution), dtype=tx_signal.dtype)

    # Create interpolator for signal
    sample = np.arange(tx_signal.size)
    fun = interpolate.interp1d(
        sample,
        tx_signal,
        kind=kind,
        bounds_error=False,
        fill_value=0,
    )

    if fir_filter == FIRFilter.b414d15_gaus:
        filt = apply_b414d15_gaus
        # TODO: i just magically know this filter chain is decimated by 15 - maybe this could be
        # better structured
        decimation = 15
    elif fir_filter == FIRFilter.mu2004:
        filt = mu_radar_filter_post_2004
        decimation = 120
    else:
        # TODO: finish this
        raise ValueError("todo error here")

    super_sample = np.arange(tx_signal.size * decimation) / decimation

    # Signal value of tx sub resolutions
    offsets = np.linspace(0, 1, sub_resolution, endpoint=False)
    for ind in range(sub_resolution):
        x = fun(super_sample - offsets[ind])
        modulated_tx[tx_stencil, ind] = filt(x)[tx_stencil]

    return modulated_tx


def simulate_pulse_code(
    code: npt.NDArray[np.float64],
    baud_length_usec: int,
    t_samp_usec: int | float,
    ipp_t_usec: int,
    signal_length: int,
    start_samp: int | float = 0,
) -> npt.NDArray[np.complex128]:
    t_usec = (np.arange(signal_length) - start_samp) * t_samp_usec

    t_in_ipp_usec = t_usec % ipp_t_usec
    t_ind = (t_in_ipp_usec // baud_length_usec).astype(np.int64)
    signal = np.zeros(t_usec.shape, dtype=np.complex128)
    inds = np.logical_and(t_in_ipp_usec >= 0, t_in_ipp_usec <= baud_length_usec * len(code))
    signal[inds] = code[t_ind[inds]]
    return signal


def tx_signal_model(
    code: npt.NDArray[np.float64],
    baud_length_usec: int,
    t_samp_usec: int,
    tx_start_samp: int,
    ipp_samps: int,
    read_length: int,
    bandwidth: float,
    start_samp: int = 0,
    sub_resolution: int = 1,
    fir_filter: FIRFilter = FIRFilter.b414d15_gaus,
) -> npt.NDArray[np.complex128]:
    """
    Tx signal simulation, based on a filtered version of a analytic coded finite bandwidth signal.

    Args:
        code: Transmitted code
        baud_length_usec: Transmission baud length
        t_samp_usec: Receiver sample time
        tx_start_samp: tx start sample relative to the inter pulse period
        start_samp: start sample relative to the inter pulse period, the generated data will start at this point.
        read_length: How many samples to generate
        ipp_samps: interpulse period samples
        sub_resolution: Datapoints per sample to use when upsampling the tx signal
        kind: interpolation kind {'linear', 'nearest', 'nearest-up', 'zero',
            'slinear', 'quadratic', 'cubic', 'previous', 'next'}

    Returns:
        Tx signal of size (read_length, sub resolution) containing the interpolated signal based on the code
    """
    # TODO: update docstring

    ipp_t_usec = ipp_samps * t_samp_usec
    if fir_filter == FIRFilter.b414d15_gaus:
        filt = apply_b414d15_gaus
        decimation = 15
    elif fir_filter == FIRFilter.mu2004:
        filt = mu_radar_filter_post_2004
        decimation = 120
    else:
        raise ValueError("todo error here")

    signals = np.zeros((read_length, sub_resolution), dtype=np.complex128)
    offsets = np.linspace(0, 1, sub_resolution)

    for ind in range(sub_resolution):
        signal = simulate_pulse_code(
            code=code,
            baud_length_usec=baud_length_usec,
            t_samp_usec=t_samp_usec / decimation,
            ipp_t_usec=ipp_t_usec,
            signal_length=read_length * decimation,
            start_samp=(start_samp + offsets[ind]) * decimation,
        )

        # filter to the initial bandwidth of the transmitter
        spectrum = fft(signal)
        freqs = fftfreq(read_length * decimation, d=decimation * 1e6 / t_samp_usec)
        mask = np.abs(freqs) <= bandwidth / 2
        filtered_spectrum = spectrum * mask
        fsignal = ifft(filtered_spectrum).real

        # filter according to the receiver chain
        signals[:, ind] = filt(fsignal)

    # Signal value of tx_indices
    norms = np.sum(np.conj(signals) * signals, axis=0)
    signals = signals / norms[None, :]

    return signals
