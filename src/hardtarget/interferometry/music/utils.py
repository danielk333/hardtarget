import numpy as np
import numpy.typing as npt


def correlation_matrix(rx_per_channel: npt.NDArray) -> npt.NDArray:
    """
    Calculate correlation matrix over rx samples per channel

    Args:
        rx_per_channel: NxM matrix where N is amount of channels, M is amount of samples
    Return:
        (1/M)*(rx_per_channel*rx_per_channel')
    """

    return np.dot(rx_per_channel, rx_per_channel.conj().T) / rx_per_channel.shape[1]
