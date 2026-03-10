"""Discrete Polynomial-phase Transform, or DPT type specifics"""

from dataclasses import dataclass

from hardtarget.target_estimation.types import ExtendedTargetEstimationProParams, TargetEstimationCfgParams


@dataclass(frozen=True)
class DPTCfgParams(TargetEstimationCfgParams):
    """DPT specific configuration parameters, can be configured under the dpt section in the .ini file

    Args:
        ipp_delay_parameter: How many IPPs to use for estimating the numerical derivative of the phase of
                             the coherent signal if using the polynomial phase transform method.
                             Also used by the Fast GMF method to calculate the acceleration resolution to
                             sample. A larger number means smaller accelerations can be determined at higher
                             resolution. Note this can never be less or equal then the n_ipp param.

    """

    ipp_delay_parameter: int = 10


@dataclass(frozen=True)
class DPTProParams(ExtendedTargetEstimationProParams):
    """
    DPT specific process parameters
    """

    decimated_ipp_delay_parameter: int = 1
