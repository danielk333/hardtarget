import scipy.constants as constants
import numpy as np


LUNAR_DISTANCE = 3.84399e8  # m
EARTH_RADIUS = 6.3781e6  # m


def unit_to_SI(val, unit):
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


def SI_to_unit(val, unit):
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


def unit_to_range_gate(val, unit, sample_rate):
    if unit == "sample":
        return val
    val = unit_to_SI(val, unit)
    val = sample_rate*val/constants.c - 1
    return np.round(val).astype(np.int64)


def range_gate_to_unit(val, unit, sample_rate):
    if unit == "sample":
        return val
    val = constants.c*val/sample_rate + 1
    return SI_to_unit(val, unit)
