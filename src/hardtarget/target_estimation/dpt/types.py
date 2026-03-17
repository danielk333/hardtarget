"""Discrete Polynomial-phase Transform, or DPT type specifics"""

from dataclasses import dataclass

from hardtarget.target_estimation.types import ExtendedTargetEstimationProParams, TargetEstimationCfgParams


@dataclass(frozen=True)
class DPTCfgParams(TargetEstimationCfgParams):
    """DPT specific configuration parameters, can be configured under the dpt section in the .ini file

    Args:
        ipp_delay_parameter: How many IPPs to use for estimating the numerical derivative of the phase of
                             the coherent signal if using the polynomial phase transform method.
                             A larger number means smaller accelerations can be determined at higher
                             resolution. Note this can never be larger or equal to the n_ipp param.

    """

    ipp_delay_parameter: int = 1


@dataclass(frozen=True)
class DPTProParams(ExtendedTargetEstimationProParams):
    """
    DPT specific process parameters
    """

    decimated_ipp_delay_parameter: int = 1
