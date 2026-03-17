from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types import CfgParams, ProParams


@dataclass(frozen=True)
class DOACfgParams(CfgParams):
    elevation_limit: int = 0
    resolution: int = 25


@dataclass(frozen=True)
class DOAProParams(ProParams):
    kx: npt.NDArray = field(default_factory=lambda: np.zeros((1, 1)))
    ky: npt.NDArray = field(default_factory=lambda: np.zeros((1, 1)))
    kz: npt.NDArray = field(default_factory=lambda: np.zeros((1, 1)))


class DOAVars(NamedTuple):
    vals: npt.NDArray[np.complex64]  # tmp for debugging
    k_vec: npt.NDArray[np.float32]
    peak: npt.NDArray[np.complex64] | np.complex64
    azimuth: npt.NDArray[np.float32]
    elevation: npt.NDArray[np.float32]


DOAOutArgs = DOAVars
