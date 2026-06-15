"""
Tx signal models, used to simulate tx signals when not available.
"""

import numpy as np
import numpy.typing as npt
import scipy.interpolate as interpolate
import scipy.signal as sc_signal

def tx_modulation_model(
    tx_signal: npt.NDArray[np.complex128],
    tx_stencil: npt.NDArray[np.bool],
    out_sample_rate: int,
    in_sample_rate: int,
    frequency_cutoff: float,
    sub_resolution: int = 1,
    kind: str = "linear",
) -> npt.NDArray[np.complex128]:
    modulated_tx = np.empty((tx_signal.size, sub_resolution), dtype=tx_signal.dtype)

    # Create interpolator for signal
    sample = np.arange(tx_signal.size)
    fun = interpolate.interp1d(
        sample[tx_stencil],
        tx_signal[tx_stencil],
        kind=kind,
        bounds_error=False,
        fill_value=0,
    )

    super_rate = int(in_sample_rate / out_sample_rate)
    super_sample = np.arange(tx_signal.size * super_rate) / super_rate

    # Signal value of tx sub resolutions
    offsets = np.linspace(0, 1, sub_resolution, endpoint=False)
    for ind in range(sub_resolution):
        x = fun(super_sample - offsets[ind])
        numtaps = super_rate + 1
        # TODO: which filter is maybe input variable?
        fir = sc_signal.firwin(numtaps, frequency_cutoff, fs=in_sample_rate)
        x_filt = sc_signal.lfilter(fir, 1.0, x)
        modulated_tx[tx_stencil, ind] = x_filt[::super_rate][tx_stencil]

    return modulated_tx


def tx_signal_model(
    code: npt.NDArray[np.float64],
    baud_length_usec: int,
    t_samp_usec: int,
    tx_start_samp: int,
    ipp_samps: int,
    read_length: int,
    start_samp: int = 0,
    sub_resolution: int = 1,
    kind: str = "linear",
) -> npt.NDArray[np.complex128]:
    """
    Tx signal simulation, based on the signal code a interpolated model estimates the
    tx signal.

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

    # If the reciver side is oversampling the code needs to be upsampled
    transmitted_code_size = len(code)
    upsample_scale = baud_length_usec // t_samp_usec
    received_code_size = transmitted_code_size * upsample_scale
    if received_code_size > transmitted_code_size:
        code = np.repeat(code, upsample_scale)

    # Zero pad code to not miss any start/end shifts
    code = np.concatenate([np.array([0]), code, np.array([0])])
    # Adjust startsample according to the padding.
    start_samp += 1

    # Create interpolator for code
    sample = np.arange(tx_start_samp, tx_start_samp + len(code))
    fun = interpolate.interp1d(
        sample,
        code,
        kind=kind,
        bounds_error=False,
        fill_value=0,
    )

    # Array of tx indicies
    tx_indicies = np.repeat(
        np.arange(start_samp, start_samp + read_length).reshape((read_length, 1)),
        sub_resolution,
        axis=1,
    ) % ipp_samps + np.linspace(0, 1, sub_resolution, endpoint=False)

    # Signal value of tx_indices
    tx = fun(tx_indicies)
    tx /= np.sum(np.conj(tx) * tx, axis=0)

    return tx
