from typing import Optional

import numpy as np
import numpy.typing as npt

from hardtarget.interferometry.types import DOAProParams, DOAVars


def plot_direction_of_arrival(
    axes: npt.NDArray,  # of type Axes
    out_data: DOAVars,
    pro: DOAProParams,
    limit: Optional[float] = None,
) -> npt.NDArray:  # of type Axes:
    """
    Plots direction of arrival results, axes needs to be of atleast size (2,2).
    """

    if limit:
        detections = out_data.peak >= limit
    else:
        detections = np.zeros(out_data.peak.shape, dtype=np.bool)

    # Plot azimut and elvation over time
    axes[0, 0].plot(out_data.elevation)
    axes[0, 0].plot(np.arange(0, len(out_data.elevation))[detections], out_data.elevation[detections], ".r")
    axes[0, 0].set_ylabel("degrees")
    axes[0, 0].set_title("elevation")

    axes[0, 1].plot(out_data.azimuth)
    axes[0, 1].plot(np.arange(0, len(out_data.azimuth))[detections], out_data.azimuth[detections], ".r")
    axes[0, 1].set_ylabel("degrees")
    axes[0, 1].set_title("azimuth")

    # Plot objects movement over time
    axes[1, 0].plot(out_data.k_vec[:, 0], out_data.k_vec[:, 1], ".")
    axes[1, 0].plot(out_data.k_vec[detections, 0], out_data.k_vec[detections, 1], "--ro")
    axes[1, 0].set_xlim(pro.kx.min(), pro.kx.max())
    axes[1, 0].set_ylim(pro.ky.min(), pro.ky.max())
    axes[1, 0].set_title("Object path over field of view")

    # Optimal value
    axes[1, 1].plot(np.abs(out_data.peak))
    axes[1, 1].set_title("Optimized value per ipp")

    return axes
