from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from pyant.models.array import Array, ArrayParams
from scipy import optimize
from spacecoords.spherical import cart_to_sph

from hardtarget.interferometry.types import DOACfgParams, DOAProParams, DOAVars
from hardtarget.types.types import ExpParams

from .utils import correlation_matrix


def grid_search_numpy(
    rx_per_channel: npt.NDArray,
    exp: ExpParams,
    cfg: DOACfgParams,
    pro: DOAProParams,
    beam: Array,
    beam_params: ArrayParams,
) -> DOAVars:
    """
    TODO: update for simultaneous meteors

    Args:
        rx_per_channel: NxM matrix where N is amount of channels, M is amount of samples
    """

    # Calculate R matrix
    r_matrix = correlation_matrix(rx_per_channel[:, pro.rel_rgs])

    # Calculate eigen values and vector
    eig, eig_vec = np.linalg.eig(r_matrix)

    # Locate max eigen value and remove corresponding column from the eigen vector
    eig_ind = np.argmax(np.abs(eig))
    eig_vec = np.delete(eig_vec, eig_ind, axis=1)

    def landscape_function(k: npt.NDArray) -> np.complex64:
        """
        Landscape function

        Args:
            k: x,y,z position (3,) array

        """

        if k[0] ** 2 + k[1] ** 2 <= np.cos(cfg.elevation_limit):
            a = beam.channel_signals(k, beam_params)
            return (a.conj().T @ a) / (a.conj().T @ eig_vec @ eig_vec.conj().T @ a)
        else:
            return np.complex64(0)

    # Calculate landscape values
    vals = np.zeros((pro.kx.shape[0], pro.ky.shape[0]), dtype=np.complex64)
    for x in range(pro.kx.shape[0]):
        for y in range(pro.ky.shape[0]):
            if pro.kz[x, y] != np.nan:
                k = np.array([pro.kx[x, y], pro.ky[x, y], pro.kz[x, y]])
                vals[x, y] = landscape_function(k)

    # Gradient ascent from maxima
    max_ind = np.unravel_index(vals.argmax(), vals.shape)
    k_start = np.array([pro.kx[max_ind], pro.ky[max_ind], pro.kz[max_ind]])
    min_func = lambda x: -np.abs(landscape_function(x))
    k_opt = gradient_ascent(func=min_func, k_start=k_start)
    val_opt = landscape_function(k_opt)

    sph = cart_to_sph(k_opt)

    return DOAVars(
        k_vec=k_opt,
        peak=val_opt,
        azimuth=sph[0],
        elevation=sph[1],
        vals=vals,
    )


def gradient_ascent(func: Callable[[npt.NDArray], Any], k_start: npt.NDArray) -> npt.NDArray:
    """TODO Support multiple objects? Will increase in size..."""

    k_opt = optimize.fmin(func=func, x0=k_start, disp=False)

    return k_opt
