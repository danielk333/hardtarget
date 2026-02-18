"""
Tx signal models, used to simulate tx signals when not available.
"""

import numpy as np
import numpy.typing as npt
import scipy.interpolate as interpolate


def tx_signal_model(
    code: npt.NDArray[np.float64],
    tx_start_samp: int,
    read_length: int,
    ipp_samps: int,
    start_samp: int = 0,
    sub_resolution: int = 1,
    kind: str = "linear",
) -> npt.NDArray[np.complex128]:
    """
    Tx signal simulation, based on the signal code a interpolated model estimates the
    tx signal.

    Args:
        code: Transmitted code
        tx_start_samp: tx start sample relative to the inter pulse period
        start_samp: start sample relative to the inter pulse period
        read_length: How many samples are read
        ipp_samps: interpulse period samples
        sub_resolution: Datapoints per sample to use when upsampling the tx signal
        kind: interpolation kind {'linear', 'nearest', 'nearest-up', 'zero',
            'slinear', 'quadratic', 'cubic', 'previous', 'next'}

    Returns:
        Tx signal of size (read_length, sub resolution) containing the interpolated signal based on the code
    """

    # Zero pad code to not miss any start/end shifts
    code = np.concatenate([np.array([0]), code, np.array([0])])
    sample = np.arange(tx_start_samp, tx_start_samp + len(code))  # TODO change to tx_start, tx_start+code
    fun = interpolate.interp1d(
        sample,
        code,
        kind=kind,
        bounds_error=False,
        fill_value=0,
    )

    tx = np.zeros(
        (read_length, sub_resolution),
        dtype=np.complex128,
    )

    # TODO: try to only loop over subresolution instead
    for i in range(start_samp, start_samp + read_length):
        tx[i - start_samp, :] = fun((i % (ipp_samps)) + 1 - np.linspace(0, 1, sub_resolution, endpoint=False))
    for i in range(sub_resolution):
        tx[:, i] /= np.sum(np.conj(tx[:, i]) * tx[:, i])
    return tx
