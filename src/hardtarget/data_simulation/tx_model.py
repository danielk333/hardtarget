"""
Tx signal models, used to simulate tx signals when not available.
"""

import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy.fft import fft, fftfreq, ifft

from hardtarget.constants import ReceiverChainModel
from hardtarget.data_simulation.receiver_chain import (
    DigitalReceiverChain,
    get_reciver_chain,
)


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
    sub_resolution: int | npt.NDArray[np.float64] = 1,
    kind: str = "linear",
    filt: ReceiverChainModel | DigitalReceiverChain | str = ReceiverChainModel.none,
) -> npt.NDArray[np.complex128]:
    """
    Extract subsamples from a existing tx signal

    """

    if isinstance(filt, (ReceiverChainModel, str)):
        filt = get_reciver_chain(filt)

    if isinstance(sub_resolution, int):
        offsets = np.linspace(0, 1, sub_resolution, endpoint=False)
    else:
        offsets = sub_resolution

    modulated_tx = np.zeros((tx_signal.size, len(offsets)), dtype=tx_signal.dtype)

    # Create interpolator for signal
    sample = np.arange(tx_signal.size)
    fun = interpolate.interp1d(
        sample,
        tx_signal,
        kind=kind,
        bounds_error=False,
        fill_value=0,
    )

    super_sample = np.arange(tx_signal.size) - filt.delay

    for ind in range(len(offsets)):
        x = fun(super_sample - offsets[ind])
        modulated_tx[tx_stencil, ind] = x[tx_stencil]

    return modulated_tx


def simulate_pulse_code(
    code: npt.NDArray[np.float64],
    baud_length_usec: int,
    t_samp_usec: float,
    ipp_t_usec: int,
    signal_length: int,
    start_samp: float = 0,
) -> npt.NDArray[np.complex128]:

    # I code is 2 diemensional it is assumed that the signal lenght is more than 1 ipp

    t_usec = (np.arange(signal_length) - start_samp) * t_samp_usec

    code_2d = np.atleast_2d(code)
    num_pulses, code_length = code_2d.shape

    t_in_ipp_usec = t_usec % ipp_t_usec
    t_ind = (t_in_ipp_usec // baud_length_usec).astype(np.int64)

    signal = np.zeros(t_usec.shape, dtype=np.complex128)

    pulse_ind = (t_usec // ipp_t_usec).astype(np.int64)
    inds = np.logical_and.reduce(
        (
            t_in_ipp_usec >= 0,
            t_in_ipp_usec < baud_length_usec * code_length,
            pulse_ind >= 0,
            pulse_ind < num_pulses,
        )
    )

    signal[inds] = code_2d[pulse_ind[inds], t_ind[inds]]
    return signal


def phase_flip_model_impulse(
    filt: ReceiverChainModel | DigitalReceiverChain | str,
    sample_offset: float = 0.5,
) -> npt.NDArray[np.complex128]:
    if isinstance(filt, (ReceiverChainModel, str)):
        filt = get_reciver_chain(filt)

    offsets = np.linspace(-sample_offset, sample_offset, int(filt.decimation * 2 * sample_offset))

    res = np.full((len(offsets), 3), np.nan, dtype=np.complex128)
    for ind, offset in enumerate(offsets):
        signal = np.full((10 * filt.decimation,), -1, dtype=np.complex128)
        signal[: np.floor((5 + offset) * filt.decimation).astype(np.int64)] = 1
        fsignal = filt.model(signal)

        res[ind, 0] = fsignal[6]
        res[ind, 1] = fsignal[7]
        res[ind, 2] = fsignal[5]
    return res


def phase_flip_model(
    filt: ReceiverChainModel | DigitalReceiverChain | str,
    sample_offset: float = 0.5,
) -> npt.NDArray[np.complex128]:

    if isinstance(filt, (ReceiverChainModel, str)):
        filt = get_reciver_chain(filt)

    offsets = np.linspace(-sample_offset, sample_offset, int(filt.decimation * 2 * sample_offset))

    res = np.full((len(offsets), 3), np.nan, dtype=np.complex128)
    for ind, offset in enumerate(offsets):
        signal = np.full((10 * filt.decimation,), -1, dtype=np.complex128)
        signal[: np.floor((5 + offset) * filt.decimation).astype(np.int64)] = 1
        fsignal = filt.model(signal)

        res[ind, 0] = fsignal[6]
        res[ind, 1] = fsignal[7]
        res[ind, 2] = fsignal[5]
    return res


def tx_signal_model(
    code: tuple[float] | tuple[tuple[float, ...], ...] | npt.NDArray[np.float64],
    baud_length_usec: int,
    t_samp_usec: int,
    ipp_samps: int,
    read_length: int,
    bandwidth: float | None,
    start_samp: int = 0,
    sub_resolution: int | npt.NDArray[np.float64] = 1,
    filt: ReceiverChainModel | DigitalReceiverChain | str = ReceiverChainModel.b414d15_gaus,
    normalize: bool = False,
) -> npt.NDArray[np.complex128]:
    """
    Tx signal simulation, based on a filtered version of a analytic coded finite bandwidth signal.

    Args:
        code: Transmitted code, single array for one code, multiple arrays if varying per ipp
        baud_length_usec: Transmission baud length
        t_samp_usec: Receiver sample time
        ipp_samps: interpulse period samples
        read_length: How many samples to generate
        bandwith: Bandwith of transmitter
        start_samp: start sample relative to the inter pulse period, the generated data will start at this point.
        sub_resolution: Datapoints per sample to use when upsampling the tx signal
        fir_filter: Downsampling filter used on transmitter
        normalize: Normalize the output signal (z-score transform)

    Returns:
        Tx signal of size (read_length, sub resolution) containing the interpolated signal based on the code
    """
    # TODO: update docstring

    if isinstance(filt, (ReceiverChainModel, str)):
        filt = get_reciver_chain(filt)

    ipp_t_usec = ipp_samps * t_samp_usec

    if isinstance(code, tuple):
        code = np.array(code).astype(np.float64)

    if isinstance(sub_resolution, int):
        offsets = np.linspace(0, 1, sub_resolution, endpoint=False)
    else:
        offsets = sub_resolution

    signals = np.zeros((read_length, len(offsets)), dtype=np.complex128)

    for ind in range(len(offsets)):
        signal = simulate_pulse_code(
            code=code,
            baud_length_usec=baud_length_usec,
            t_samp_usec=t_samp_usec / filt.decimation,
            ipp_t_usec=ipp_t_usec,
            signal_length=int((read_length + filt.delay) * filt.decimation),
            start_samp=(start_samp + filt.delay + offsets[ind]) * filt.decimation,
        )
        # filter to the initial bandwidth of the transmitter
        if bandwidth is not None:
            spectrum = fft(signal)
            freqs = fftfreq(len(signal), d=filt.decimation * 1e6 / t_samp_usec)
            mask = np.abs(freqs) <= bandwidth / 2
            filtered_spectrum = spectrum * mask
            fsignal = ifft(filtered_spectrum)
            inds = np.abs(signal) > 0
            signal[inds] = fsignal[inds]

        # filter according to the receiver chain
        signals[:, ind] = filt.model(signal)[filt.delay :]

    if normalize:
        mu = np.mean(signals, axis=0)
        sig = np.std(signals, axis=0)
        signals = (signals - mu[None, :]) / sig[None, :]

    return signals
