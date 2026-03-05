from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types.types import CfgParams, ProParams


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


# TODO: Delete, should be moved to metecho
class XCorrOutArgs(NamedTuple):
    # SHould these first ones be removed?
    max_pow_per_delay: npt.NDArray[np.complex128]
    max_pow_per_delay_norm: npt.NDArray[np.complex128]
    best_peaks: npt.NDArray[np.complex128]
    best_dopplers: npt.NDArray[np.integer]
    max_pow_inds: npt.NDArray[np.integer]
    tot_pow: npt.NDArray[np.floating]
    # extras
    doppler_window: npt.NDArray[np.integer]
    start_window: npt.NDArray
    doppler_std: npt.NDArray[np.integer]
    start_std: npt.NDArray
    doppler_coherrence: float
    start_coherrence: float
    gauss_noise: dict
    filter_indices: npt.NDArray[np.bool]
    beast_peak_filt: npt.NDArray[np.complex128]
    tot_pow_filt: npt.NDArray[np.complex128]
    epoch: float  # seconds since epoch
