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
        snr = (np.sqrt(gmf_values) - np.sqrt(noise_floor[None, :])) ** 2 / noise_floor[None, :]
    else:
        inds = np.logical_and(range_gates >= 0, range_gates < len(noise_floor))
        snr = np.full_like(gmf_values, np.nan)
        snr[inds] = (np.sqrt(gmf_values[inds]) - np.sqrt(noise_floor[range_gates[inds]])) ** 2 / noise_floor[
            range_gates[inds]
        ]
    if dB:
        return 10 * np.log10(snr)
    else:
        return snr
