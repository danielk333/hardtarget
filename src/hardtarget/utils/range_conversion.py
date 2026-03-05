"""
Range conversion tools
"""

import numpy as np
import scipy.constants as constants

LUNAR_DISTANCE = 3.84399e8  # m
EARTH_RADIUS = 6.3781e6  # m


def unit_to_SI(val: float | int, unit: str) -> float:
    """Converts unit value to SI value {"m"/"km"/"r_e"/"ld"/"au"}"""
    if unit == "m":
        pass
    elif unit == "km":
        val *= 1e3
    elif unit == "r_e":
        val *= EARTH_RADIUS
    elif unit == "ld":
        val *= LUNAR_DISTANCE
    elif unit == "au":
        val *= constants.au
    else:
        raise ValueError(f"Unit '{unit}' not recognized, see cli description")
    return val


def SI_to_unit(val: float | int, unit: str) -> float:
    """Converts SI value to unit value {"m"/"km"/"r_e"/"ld"/"au"}"""

    if unit == "m":
        pass
    elif unit == "km":
        val /= 1e3
    elif unit == "r_e":
        val /= EARTH_RADIUS
    elif unit == "ld":
        val /= LUNAR_DISTANCE
    elif unit == "au":
        val /= constants.au
    else:
        raise ValueError(f"Unit '{unit}' not recognized, see cli description")
    return val


def unit_to_range_gate(val: float | int, unit: str, sample_rate: int | float) -> np.int64:
    """Converts unit value to range gate

    Args:
        val: value
        unit: si unit {"m"/"km"/"r_e"/"ld"/"au"}
        sample_rate: sample rate
    """

    if unit == "sample":
        return np.int64(val)
    val = unit_to_SI(val, unit)
    val = sample_rate * val / constants.c - 1
    return np.round(val).astype(np.int64)


def range_gate_to_unit(val: float | int, unit: str, sample_rate: int | float) -> float:
    """Converts unit value to range gate

    Args:
        val: value
        unit: si unit {"m"/"km"/"r_e"/"ld"/"au"}
        sample_rate: sample rate
    """
    if unit == "sample":
        return val
    val = constants.c * val / sample_rate + 1
    return SI_to_unit(val, unit)
