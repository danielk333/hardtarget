from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from pyant.models.array import Array, ArrayParams
from scipy import optimize
from spacecoords.spherical import cart_to_sph

from hardtarget.interferometry.types import DOACfgParams, DOAProParams, DOAVars
from hardtarget.types import ExpDef

from .utils import correlation_matrix


@np.errstate(all="raise")
def landscape_function(
    k: npt.NDArray, eig_vec: npt.NDArray, beam: Array, beam_params: ArrayParams
) -> npt.NDArray[np.complex64] | np.complex64:
    """
    Landscape function

    Args:
        k: x,y,z position (3,N) array or (3,) array
    Returns:
        Scalar value or (N,) array
    """
    if k.ndim > 1:
        res = np.zeros((k.shape[1],), dtype=np.complex64)
        a = beam.channel_signals(k, beam_params)
        for i in range(k.shape[1]):
            try:
                res[i] = (a[:, i].conj().T @ a[:, i]) / (
                    a[:, i].conj().T @ eig_vec @ eig_vec.conj().T @ a[:, i]
                )
            except FloatingPointError:
                res[i] = 0
        return res
    else:
        a = beam.channel_signals(k, beam_params)
        try:
            return (a.conj().T @ a) / (a.conj().T @ eig_vec @ eig_vec.conj().T @ a)
        except FloatingPointError:
            return np.complex64(0)


def grid_search_numpy(
    rx_per_channel: npt.NDArray,
    exp: ExpDef,
    cfg: DOACfgParams,
    pro: DOAProParams,
    beam: Array,
    beam_params: ArrayParams,
) -> DOAVars:
    """
    TODO: update for simultaneous meteors


    Go through each combination of x and y to determine the landscape value.
    From this locate the maxima on the landscape and use gradient descent to
    find the true maxima.

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

    # Calculate landscape values the fast non understandable way

    # Filter out Nan values
    nan_filter = np.invert(np.isnan(pro.kz))
    k_vec = np.vstack(
        [pro.kx[nan_filter].flatten(), pro.ky[nan_filter].flatten(), pro.kz[nan_filter].flatten()]
    )
    k_index = np.array(pro.k_index[nan_filter].flatten().tolist())

    # Filter out positions outside of elevation limit
    elevation_filter = (k_vec[0] ** 2 + k_vec[1] ** 2) <= np.cos(cfg.elevation_limit)
    k_vec = k_vec[:, elevation_filter]
    k_index = k_index[elevation_filter]

    # Calculate peaks
    vals = np.zeros((pro.kx.shape[0], pro.ky.shape[0]), dtype=np.complex64)
    vals[k_index[:, 0], k_index[:, 1]] = landscape_function(
        k=k_vec,
        eig_vec=eig_vec,
        beam=beam,
        beam_params=beam_params,
    )

    # Extract N peaks to run gradient ascent from
    peak_inds = get_distributed_peaks(vals, cfg.distributed_peaks)
    k_peaks = np.array(
        [
            pro.kx[peak_inds[:, 0], peak_inds[:, 1]],
            pro.ky[peak_inds[:, 0], peak_inds[:, 1]],
            pro.kz[peak_inds[:, 0], peak_inds[:, 1]],
        ]
    ).T

    # Gradient ascent
    val_opt = np.complex64(0)
    k_opt = np.empty((3,), dtype=np.float32)
    for k_start in k_peaks:
        k_vec, val = gradient_ascent(
            func=lambda k: landscape_function(
                k=k,
                eig_vec=eig_vec,
                beam=beam,
                beam_params=beam_params,
            ),
            k_start=k_start,
        )
        if val > val_opt:
            val_opt = val
            k_opt = k_vec

    # Convert to spherical coordinates
    sph = cart_to_sph(k_opt, degrees=True)

    return DOAVars(
        k_vec=k_opt,
        peak=val_opt,
        azimuth=sph[0],
        elevation=sph[1],
    )


def gradient_ascent(func: Callable[[npt.NDArray], Any], k_start: npt.NDArray) -> tuple[npt.NDArray, Any]:
    """
    Args:
        func: function to maximize that accepts a k vector as input.
        k_start: start vector to start the gradient ascent from
    Returns:
        tuple containing the optimal k vector and the value at this optimal k

    """

    k_opt = optimize.fmin(func=lambda x: -np.abs(func(x)), x0=k_start, disp=False)

    return k_opt, func(k_opt)


def get_distributed_peaks(vals: npt.NDArray, n_peaks: int) -> npt.NDArray:
    """
    Get a set of distributed peaks in the vals array.
    A maximum value is taken as the first peak, all values within a 5% radius will after this be neglected.
    From the remaining values a new peak is taken, this is to enforce more exploration.

    Args:
        vals: Values to search for peaks within.
        n_peaks: Amount of peaks to return
    Returns:
        Array of (n_peaks,2) containing indexes of the n distributed peaks
    """

    if n_peaks <= 1:
        return np.array([np.unravel_index(vals.argmax(), vals.shape)])

    vals_in = vals.copy()
    # If a peak is found remove all other nearby objects to get a more distributed dataset
    deformation_radius = int(vals_in.shape[0] / 20)

    distributed_peaks = []
    for i in range(n_peaks):
        max_peak = np.unravel_index(vals_in.argmax(), vals_in.shape)
        vals_in[
            max_peak[0] - deformation_radius : max_peak[0] + deformation_radius,
            max_peak[1] - deformation_radius : max_peak[1] + deformation_radius,
        ] = 0
        distributed_peaks.append(max_peak)

    return np.array(distributed_peaks)
