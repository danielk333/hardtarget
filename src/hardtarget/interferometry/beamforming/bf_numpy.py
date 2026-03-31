import numpy as np
import numpy.typing as npt
from pyant.models.array import Array, ArrayParams
from spacecoords.spherical import cart_to_sph

from hardtarget.interferometry.types import DOACfgParams, DOAProParams, DOAVars
from hardtarget.types import ExpDef


def grid_search_numpy(
    rx: npt.NDArray,
    exp: ExpDef,
    cfg: DOACfgParams,
    pro: DOAProParams,
    beam: Array,
    beam_params: ArrayParams,
) -> DOAVars:

    vals = np.zeros((pro.kx.shape[0], pro.ky.shape[0]), dtype=np.complex64)

    for x in range(pro.kx.shape[0]):
        for y in range(pro.ky.shape[0]):
            if (pro.kx[x, y] ** 2 + pro.ky[x, y] ** 2) <= np.cos(cfg.elevation_limit):
                k0 = np.array([pro.kx[x, y], pro.ky[x, y], pro.kz[x, y]])
                a = beam.channel_signals(k0, beam_params)
                vals[x, y] = np.sum(rx.T * np.conjugate(a))  # is this really correct?

    max_ind = np.unravel_index(vals.argmax(), vals.shape)
    k_opt = np.array([pro.kx[max_ind], pro.ky[max_ind], pro.kz[max_ind]])

    sph = cart_to_sph(k_opt)

    return DOAVars(
        k_vec=k_opt,
        peak=vals[max_ind],
        azimuth=sph[0],
        elevation=sph[1],
    )
