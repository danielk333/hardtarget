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
    """
    max_corr: Best correlation from all doppler frequencies.
    max_corr_ind: Index of the max_corr within the correlation array for the specific doppler frequency.
    best_doppler: Doppler frequency that contained the best correlation.
    tot_pow: Sum of all readings from the raw data of the ipp.
    mean: Mean value of the raw data reading.
    std_dev: Standard deviation of the raw data reading.


    """

    max_corr: np.complex128 | npt.NDArray[np.complex128]
    max_corr_ind: np.integer | npt.NDArray[np.integer]
    best_doppler: np.integer | npt.NDArray[np.integer]
    tot_pow: float | npt.NDArray[np.floating]
    mean: np.floating | npt.NDArray[np.floating]
    std_dev: np.floating | npt.NDArray[np.floating]


class EchoSearchOutArgs(NamedTuple):
    """
    Each object is per ipp except epoch

    """

    max_corr: npt.NDArray[np.complex128]
    max_corr_ind: npt.NDArray[np.integer]
    best_doppler: npt.NDArray[np.integer]
    tot_pow: npt.NDArray[np.floating]
    mean: npt.NDArray[np.floating]
    std_dev: npt.NDArray[np.floating]
    epoch_us: int
