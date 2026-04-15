from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types import CfgParams, ProParams


@dataclass(frozen=True)
class EchoSearchCfgParams(CfgParams):
    doppler_freq_min: int = -30000
    doppler_freq_max: int = 5000
    doppler_freq_step: int = 1000


@dataclass(frozen=True)
class EchoSearchProParams(ProParams):
    doppler_freq_size: int = 0


class EchoSearchVars(NamedTuple):
    max_pow: np.complex128 | npt.NDArray[np.complex128]
    max_pow_norm: np.complex128 | npt.NDArray[np.complex128]
    max_peak: np.complex128 | npt.NDArray[np.complex128]
    max_pow_ind: np.integer | npt.NDArray[np.integer]
    best_doppler: np.integer | npt.NDArray[np.integer]
    ipps_pow: float | npt.NDArray[np.floating]


class EchoSearchOutArgs(NamedTuple):
    max_pow: npt.NDArray[np.complex128]
    max_pow_norm: npt.NDArray[np.complex128]
    max_peak: npt.NDArray[np.complex128]
    max_pow_ind: npt.NDArray[np.integer]
    best_doppler: npt.NDArray[np.integer]
    ipps_pow: npt.NDArray[np.floating]
