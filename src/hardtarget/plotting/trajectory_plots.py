import datetime as dt
from typing import Optional

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from radardef.types.types import ExpDef
from spacecoords import spherical

from hardtarget.data_simulation.utils import TrajectoryFunction


def plot_true_vs_estimated_trajectory(
    ax: Axes,
    exp_params: ExpDef,
    trajectory_func: TrajectoryFunction,
    doa_peaks: npt.NDArray,
    doa_azimuth: npt.NDArray,
    doa_elevation: npt.NDArray,
    ranges: npt.NDArray,
    pointing: npt.NDArray,
    measurement_start: dt.datetime,
    measurement_end: dt.datetime,
    target_start_us: int,
    target_end_us: int,
    detection_limit: Optional[float] = None,
) -> Axes:
    """
    Plot the true trajectory of object and the estimated trajectory relative to the radarstation.


    Args:
        ax: Axis with projection 3D
        exp_params: Experiment definition,
        trajectory_func: TrajectoryFunction to visualise objects trajectory
        doa_peaks: Peak values per coherent integrations during DOA analysis
        doa_azimuth: Azimuth vector for each coherent integration from DOA analysis
        doa_elevation: Elevation vector for each coherent integration from DOA analysis
        ranges: Range vector for each coherent integration from target estimation analysis
        pointing: pointing in spherical coordinates (azimuth, elevation)
        measurement_start: Start time of measurement
        measurement_end: End time of measurement
        target_start_us: target start relative to the measurement start in microseconds
        target_end_us: target end relative to the measurement start in microseconds
        detection_limit (optional): min value of doa peak to visualize, everything below this will be filtered out.
                                    If not set all datapoints will be shown.
    """
    ax.set_title("Real vs Estimated trajectory")
    measurement_length_us = (measurement_end - measurement_start).total_seconds() * 1e6
    t_us = np.arange(measurement_length_us, step=exp_params.t_samp_usec)
    full_trajectory = trajectory_func(t_us * 1e-6)

    t_target_us = np.arange(target_start_us, stop=target_end_us, step=exp_params.t_samp_usec)
    noticable_trajectory = trajectory_func(t_target_us * 1e-6)

    # Trajectory of object
    ax.plot(
        full_trajectory[0],
        full_trajectory[1],
        full_trajectory[2],
        "--b",
        label=" True trajectory",
    )
    ax.plot(
        noticable_trajectory[0],
        noticable_trajectory[1],
        noticable_trajectory[2],
        "r",
        label="Measured trajectory",
    )

    # Radar station location is at center:
    radar_loc = [0, 0, 0]
    ax.plot(radar_loc[0], radar_loc[1], radar_loc[2], ".g", label="Radar station")

    # Pointing direction
    pointing_cart = spherical.sph_to_cart(
        np.array([pointing[0], pointing[1], 1]),
        degrees=True,
    ).round(decimals=10)

    pointing_cart *= np.max(ranges)
    ax.plot(
        [0, pointing_cart[0]], [0, pointing_cart[1]], [0, pointing_cart[2]], "-y", label="Pointing direction"
    )
    if detection_limit:
        doa_detections = doa_peaks > detection_limit
    else:
        doa_detections = np.ones(doa_peaks.shape, dtype=np.bool)

    estimated_trajectory = spherical.sph_to_cart(
        np.vstack([doa_azimuth[doa_detections], doa_elevation[doa_detections], ranges[doa_detections]]),
        degrees=True,
    )

    ax.plot(
        estimated_trajectory[0],
        estimated_trajectory[1],
        estimated_trajectory[2],
        ".m",
        label="Estimated trajectory",
    )

    return ax
