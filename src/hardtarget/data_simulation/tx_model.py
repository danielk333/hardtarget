"""
Tx signal models, used to simulate tx signals when not available.
"""

import numpy as np
import numpy.typing as npt
import scipy.interpolate as interpolate


def tx_signal_model(
    code: npt.NDArray[np.float64],
    ipp_samps: int,
    read_length: int,
    start_sample: int = 0,
    sub_resolution: int = 1,
    kind: str = "linear",
) -> npt.NDArray[np.complex128]:
    """
    Tx signal simulation, based on the signal code a interpolated model estimates the
    tx signal.

    Args:
        code: Transmitted code
        ipp_samps: interpulse period samples
        read_length: How many samples are read
        start_sample: start sample
        sub_resolution: Datapoints per sample to use when upsampling the tx signal
        kind: interpolation kind {'linear', 'nearest', 'nearest-up', 'zero',
            'slinear', 'quadratic', 'cubic', 'previous', 'next'}

    Returns:
        Tx signal of size (read_length, sub resolution) containing the interpolated signal based on the code
    """

    # Zero pad code to not miss any start/end shifts
    code = np.concatenate([np.array([0]), code, np.array([0])])
    sample = np.arange(0, len(code))
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

    start = start_sample % ipp_samps
    stop = read_length + (start_sample % ipp_samps)
    # TODO: try to only loop over subresolution instead
    for i in range(start, stop):
        tx[i, :] = fun((i % (ipp_samps)) + 1 - np.linspace(0, 1, sub_resolution, endpoint=False))
    for i in range(sub_resolution):
        tx[:, i] /= np.sum(np.conj(tx[:, i]) * tx[:, i])
    return tx
