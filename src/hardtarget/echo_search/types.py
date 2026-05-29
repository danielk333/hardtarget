from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types import CfgParams, ProParams


@dataclass(frozen=True)
class EchoSearchCfgParams(CfgParams):
    """
    corr_filter_limit:  limit to remove coherrent goals for noise calculations
    """

    doppler_freq_min: int = -25000
    doppler_freq_max: int = 25000
    doppler_freq_step: int = 1000
    corr_filter_limit: float = 0.5


@dataclass(frozen=True)
class EchoSearchProParams(ProParams):
    doppler_frequencies: npt.NDArray[np.int32] = field(default_factory=lambda: np.zeros((1,), dtype=np.int32))


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
    max_corr_delay: np.integer | npt.NDArray[np.integer]
    best_doppler: np.integer | npt.NDArray[np.integer]
    tot_pow: float | npt.NDArray[np.floating]
    mean: np.floating | npt.NDArray[np.floating]
    std_dev: np.floating | npt.NDArray[np.floating]


class EchoSearchOutArgs(NamedTuple):
    """
    Each object is per ipp except epoch

    max_corr: Best correlation from all doppler frequencies.
    max_corr_ind: Index of the max_corr within the correlation array for the specific doppler frequency.
    best_doppler: Doppler frequency that contained the best correlation.
    tot_pow: Sum of all readings from the raw data of the ipp.
    mean: Mean value of the raw data reading.
    std_dev: Standard deviation of the raw data reading.
    epoch_us: Data timepoint of analysis start

    """

    max_corr: npt.NDArray[np.complex128]
    max_corr_ind: npt.NDArray[np.integer]
    max_corr_delay: npt.NDArray[np.integer]
    best_doppler: npt.NDArray[np.integer]
    tot_pow: npt.NDArray[np.floating]
    mean: npt.NDArray[np.floating]
    std_dev: npt.NDArray[np.floating]
    epoch_us: int
