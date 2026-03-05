"""General Matched Filter Optimization type specifics"""

from dataclasses import dataclass

from hardtarget.types.types import CfgParams, ProParams


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
