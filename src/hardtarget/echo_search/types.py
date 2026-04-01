from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types import CfgParams, ProParams


@dataclass(frozen=True)
class XCorrCfgParams(CfgParams):
    doppler_freq_min: int = -30000
    doppler_freq_max: int = 5000
    doppler_freq_step: int = 1000


@dataclass(frozen=True)
class XCorrProParams(ProParams):
    doppler_freq_size: int = 0


class XCorrVariables(NamedTuple):
    max_pow: np.complex128 | npt.NDArray[np.complex128]
    max_pow_norm: np.complex128 | npt.NDArray[np.complex128]
    max_peak: np.complex128 | npt.NDArray[np.complex128]
    max_pow_ind: np.integer | npt.NDArray[np.integer]
    best_doppler: np.integer | npt.NDArray[np.integer]
    ipps_pow: float | npt.NDArray[np.floating]


XCorrOutArgs = XCorrVariables
