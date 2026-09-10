from typing import Optional

import numpy as np
import numpy.typing as npt


def snr(
    gmf_values: npt.NDArray[np.floating | np.integer],
    noise_floor: npt.NDArray,
    range_gates: Optional[npt.NDArray[np.integer]] = None,
    dB: bool = False,
) -> npt.NDArray:
    """Convert matched filter value to SNR based on range dependant noise floor
    (assumes MF has dimensions (N,range) unless range_gates is given)
    """
    if range_gates is None:
        snr = (gmf_values - noise_floor[None, :]) / noise_floor[None, :]
    else:
        inds = np.logical_and(range_gates >= 0, range_gates < len(noise_floor))
        snr = np.full_like(gmf_values, np.nan)
        snr[inds] = (gmf_values[inds] - noise_floor[range_gates[inds]]) / noise_floor[range_gates[inds]]
    if dB:
        return 10 * np.log10(snr)
    else:
        return snr
