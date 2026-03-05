"""Preprocessing tools"""

import numpy as np
import numpy.typing as npt


def filter_low_tx_signal(
    rx_window_indices: npt.NDArray[np.int32],
    tx: npt.NDArray[np.complexfloating],
    tx_power_limit: float = 1e-10,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.complexfloating]]:
    """A filter function that removes all samples with too low tx-signal power
    to matter for the matched filter, thereby reducing number of operations needed.
    """
    tx_p = np.real(tx * np.conj(tx))
    inds = tx_p > tx_power_limit
    return rx_window_indices[inds], tx[inds]
