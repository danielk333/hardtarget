import numpy as np
import numpy.typing as npt

from hardtarget.event_detection.types import XCorrOutArgs


def plot_event_detection(
    axes: npt.NDArray,
    out_data: XCorrOutArgs,
    limit: float = 0.5,
) -> npt.NDArray:  # of type axes
    """
    Plot result from event detection, axis of size (2,2) is a must
    """

    # filter = out_data.max_peak > limit

    axes[0, 0].plot(np.abs(out_data.max_peak))
    axes[0, 0].set_ylabel("Max peak")
    # Normalised power
    axes[0, 1].plot(np.abs(out_data.max_pow_norm))
    axes[0, 1].set_ylabel("max_power_ind")
    # Ipps power
    axes[1, 0].plot(np.abs(out_data.ipps_pow))
    axes[1, 0].set_ylabel("Ipp power")

    return axes
