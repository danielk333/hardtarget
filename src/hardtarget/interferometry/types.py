from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types import CfgParams, ProParams


@dataclass(frozen=True)
class DOACfgParams(CfgParams):
    """
    Args:
        elevation_limit: Elevation limit
        resolution: Resolution of grid search
        distributed_peaks: Number of peaks to run gradient ascent from
    """

    elevation_limit: int = 0
    resolution: int = 25
    distributed_peaks: int = 1


@dataclass(frozen=True)
class DOAProParams(ProParams):
    kx: npt.NDArray = field(default_factory=lambda: np.zeros((1, 1)))
    ky: npt.NDArray = field(default_factory=lambda: np.zeros((1, 1)))
    kz: npt.NDArray = field(default_factory=lambda: np.zeros((1, 1)))
    k_index: npt.NDArray = field(default_factory=lambda: np.zeros((1, 1)))


class DOAVars(NamedTuple):
    k_vec: npt.NDArray[np.float32]
    peak: npt.NDArray[np.complex64] | np.complex64
    azimuth: npt.NDArray[np.float32]
    elevation: npt.NDArray[np.float32]


class DOAOutArgs(NamedTuple):
    k_vec: npt.NDArray[np.float32]
    peak: npt.NDArray[np.complex64] | np.complex64
    azimuth: npt.NDArray[np.float32]
    elevation: npt.NDArray[np.float32]
    epoch_us: int
