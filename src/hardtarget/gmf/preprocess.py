import numpy as np


def filter_low_tx_signal(acc_phasors, rx_window_indices, z_tx, tx_power_limit=1e-10):
    """A filter function that removes all samples with too low tx-signal power
    to matter for the matched filter, thereby reducing number of operations needed.
    """
    tx_p = np.real(z_tx*np.conj(z_tx))
    inds = tx_p > tx_power_limit
    return acc_phasors[:, inds], rx_window_indices[inds], z_tx[inds]
