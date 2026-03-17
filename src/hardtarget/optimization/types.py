"""General Matched Filter Optimization type specifics"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types import CfgParams, ProParams


@dataclass(frozen=True)
class OptimizeCfgParams(CfgParams):
    """
    Optimize specific configuration parameters,
    can be configured under the optmize section in the .ini file.

    Args:
        path: path to analysis to optimize
    """

    path: str = "unkwn"


@dataclass(frozen=True)
class OptimizeProParams(ProParams):
    """
    Optimize specific process parameters

    Args:
        acceleration_step: Acceleration step
    """

    acceleration_step: int = 2
    il0_rx_window_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))


class MFOptimizeVariables(NamedTuple):
    """Container for compacting the variables set by the GMF Optimize function."""

    r_vec_opt: npt.NDArray[np.float64]
    v_vec_opt: npt.NDArray[np.float64]
    a_vec_opt: npt.NDArray[np.float64]
    peak_vals: npt.NDArray[np.float64]  # peak magnitude
    dc: npt.NDArray[np.float32]
    t: npt.NDArray[np.float32]


MFOptimizeOutArgs = MFOptimizeVariables


class OptStart(NamedTuple):
    r_vec: float = 0.0
    v_vec: float = 0.0
    a_vec: float = 0.0
    dc: float = 0.0
    t: float = 0.0
