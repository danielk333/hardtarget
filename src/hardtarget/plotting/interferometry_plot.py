import matplotlib.animation as animation
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hardtarget.interferometry.types import DOAProParams, DOAVars


def plot_direction_of_arrival(
    fig: Figure,
    axes: npt.NDArray,  # of type Axes
    out_data: DOAVars,
    pro: DOAProParams,
    limit: float = 10.0,
    include_animation: bool = True,
) -> tuple[npt.NDArray, animation.FuncAnimation | None]:  # of type Axes:
    """
    Plots direction of arrival results, axes needs to be of size (3,2), without animations (2,2) will suffice.
    Note, to be able to visualize the animation it needs to be stored to a local variable.

    """

    detections = out_data.peak >= limit

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
    axes[1, 0].plot(out_data.k_vec[detections, 0], out_data.k_vec[detections, 1], ".r")
    axes[1, 0].set_xlim(pro.kx.min(), pro.kx.max())
    axes[1, 0].set_ylim(pro.ky.min(), pro.ky.max())
    axes[1, 0].set_title("Object path over field of view")

    # Optimal value
    axes[1, 1].plot(np.abs(out_data.peak))
    axes[1, 1].set_title("Optimized value per ipp")

    ani = None
    if include_animation:
        # Animate vals over time
        def animate(i: int) -> tuple[Axes]:

            return (axes[2, 0].pcolormesh(pro.kx, pro.ky, np.abs(out_data.vals[i])),)

        # Note, must be saved to local var or animation wont work
        axes[2, 0].set_title("Landscape values per ipp over time")
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=out_data.vals.shape[0],
            interval=200,
            repeat_delay=2000,
            blit=True,
        )

    return axes, ani
