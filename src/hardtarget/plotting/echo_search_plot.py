from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.constants
from radardef.types.types import ExpDef

from hardtarget.echo_search.types import EchoSearchOutArgs


def plot_echo_search(
    axes: npt.NDArray,
    exp: ExpDef,
    out_data: EchoSearchOutArgs,
    convert_axis: bool = True,
    limit: Optional[float] = None,
) -> npt.NDArray:  # of type axes
    """
    Plot result from echo search, axis of size (2,2) is a must

    axes: Axes of a subplot
    exp: Experiment definition
    out_data: Echo search output
    convert_axis (optional): If axis should be converted to km and km/s
    limit (optional): limit for confirmed echo, if set data points above limit will be highlighted.
    """

    if limit:
        filter = out_data.max_corr > limit
    else:
        filter = np.zeros(out_data.max_corr.shape, dtype=np.bool)

    ipps = np.arange(out_data.max_corr.shape[0])

    # Max peak per ipp
    axes[0, 0].plot(ipps, np.abs(out_data.max_corr))
    axes[0, 0].plot(ipps[filter], np.abs(out_data.max_corr[filter]), "r.")

    axes[0, 0].set_ylabel("Max peak value")
    axes[0, 0].set_xlabel("Ipp")

    # total power per ipp
    axes[1, 0].plot(ipps, np.abs(out_data.tot_pow))
    axes[1, 0].plot(ipps[filter], np.abs(out_data.tot_pow[filter]), "r.")
    axes[1, 0].set_ylabel("Ipp power")
    axes[1, 0].set_xlabel("Ipp")

    if convert_axis:
        # Range based on max power indicies.
        range = (
            (out_data.max_corr_ind * exp.t_samp_usec + exp.t_rx_start_usec)
            * 1e-6
            * 0.5
            * scipy.constants.c
            * 1e-3
        )
        axes[0, 1].plot(ipps, range)
        axes[0, 1].plot(ipps[filter], range[filter], "r.")
        axes[0, 1].set_ylabel("Range [km]")
        axes[0, 1].set_xlabel("Ipp")

        # Doppler
        doppler_kms = (out_data.best_doppler / exp.radar_frequency) * scipy.constants.c * 1e-3
        axes[1, 1].plot(ipps, doppler_kms)
        axes[1, 1].plot(ipps[filter], doppler_kms[filter], "r.")
        axes[1, 1].set_ylabel("Doppler [km/s]")
        axes[1, 1].set_xlabel("Ipp")

    else:
        axes[0, 1].plot(ipps, out_data.max_corr_ind)
        axes[0, 1].plot(ipps[filter], out_data.max_corr_ind[filter], "r.")
        axes[0, 1].set_ylabel("Max power indices")
        axes[0, 1].set_xlabel("Ipp")

        axes[1, 1].plot(ipps, out_data.best_doppler)
        axes[1, 1].plot(ipps[filter], out_data.best_doppler[filter], "r.")
        axes[1, 1].set_ylabel("Doppler [Hz]")
        axes[1, 1].set_xlabel("Ipp")

    return axes
