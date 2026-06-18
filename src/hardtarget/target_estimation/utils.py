"""Matched filter utilities"""

import numpy as np
import numpy.typing as npt


def default_mf_vars_items(
    vector_size: tuple,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    vals = np.zeros(vector_size, dtype=np.float32)
    dc = np.zeros(vector_size, dtype=np.float32)
    v = np.zeros(vector_size, dtype=np.float32)
    a = np.zeros(vector_size, dtype=np.float32)
    phi = np.zeros(vector_size, dtype=np.float32)

    return vals, dc, v, a, phi


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
