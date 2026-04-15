from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

SIMULATION_PARAMS = [
    "epoch",
    "start_time",
    "end_time",
    "noise_sigma",
]

SIMULATION_DATA = [
    "ranges",
    "velocities",
    "accelerations",
    "snrs",
    "times",
]


# Type hinting for a common declaration of a trajectory function
class TrajectoryFunction(Protocol):
    def __call__(
        self,
        t: npt.NDArray,
    ) -> npt.NDArray: ...

    """Trajectory function

    Args:
        t: timepoints in seconds

    Returns:
        Trajectory as a (3,N timepoints) numpy array
    """


class DRFSimParams(NamedTuple):
    epoch: str | int = "2021-04-12T12:15:40"
    start_time_us: int = 0
    end_time_us: int = 10000
    target_start_time_us: int = 0
    target_end_time_us: int = 10000
    noise_sigma: float = 0
    tx_amp: float = 1


def waveform_generator(
    n_tx_samps: int,
    sample_rate: float,
    baud_length_sec: float,
    code: npt.NDArray,
    dtype: Any = np.complex64,
) -> npt.NDArray:
    """Waveform generator"""
    t = np.arange(n_tx_samps) / sample_rate
    t_ind = (t // baud_length_sec).astype(np.int64)
    signal = np.zeros(t.shape, dtype=dtype)
    inds = np.logical_and(t >= 0, t <= baud_length_sec * len(code))
    signal[inds] = code[t_ind[inds]]
    return signal


def noise_generator(noise_sigma: float, shape: tuple, dtype: Any = np.complex128) -> npt.NDArray:
    """Noise generator"""
    return noise_sigma * (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
