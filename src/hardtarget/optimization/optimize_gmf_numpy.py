"""The Numpy Implementations of the general matched filter optimization method"""

import numpy as np
import numpy.typing as npt
import scipy.constants as constants
import scipy.optimize as sco
from radardef.types import ExpParams

from hardtarget.optimization.types import OptimizeCfgParams, OptimizeProParams
from hardtarget.utils.time_conversion import ipp_time_to_sample


def optimize_gmf_np(
    tx: npt.NDArray[np.complexfloating],
    ipp: npt.NDArray,
    exp_params: ExpParams,
    cfg_params: OptimizeCfgParams,
    pro_params: OptimizeProParams,
    r_vec: float,
    v_vec: float,
    a_vec: float,
) -> tuple[float, float, float, float]:
    """
    Maximize the Generalized Matched Filter GMF value using function
    optimization in continuous variable space.

    Args:
        tx: Transmitted signal
        ipp: The entire sample vector
        exp_params: Experiment parameters
        cfg_params: Configuration parameters
        pro_params: Process parameters
        gmf_start: The r/v/a value from the previous cohhind analys

    Returns
        Optimized r_vec, v_vec, a_vec, val

    """
    sample_inds = pro_params.il0_rx_window_indices

    def neg_gmf_direct(
        x: npt.NDArray,
        r0: float,
        sample_inds: npt.NDArray[np.int32],
        wavelength: float,
        sample_rate: float,
        tx0_samp: int,
        tx: npt.NDArray[np.complexfloating],
        ipp: npt.NDArray,
    ) -> npt.NDArray:
        rg0 = np.floor((r0 / constants.c) * sample_rate).astype(np.int64) + tx0_samp
        inds = sample_inds + rg0

        sample_t = inds / sample_rate
        r = r0 + x[0] * sample_t + 0.5 * x[1] * sample_t**2.0
        phase = 2.0 * np.pi * np.mod(r / wavelength, 1)
        model_signal = tx * np.exp(-1j * phase)  # TODO: how to handle sub_resolution

        decoded_echo = ipp[inds] * model_signal

        return -(np.abs(np.sum(decoded_echo)) ** 2)

    # TODO: make so that the input parameters for minimize can be customized trough the config file
    # such as optimization limits and method

    result = sco.minimize(
        neg_gmf_direct,
        [v_vec, a_vec],
        args=(
            r_vec,
            sample_inds,
            exp_params.wavelength,
            exp_params.sample_rate,
            ipp_time_to_sample(exp_params.t_tx_start_usec, exp_params.sample_rate),
            tx,  # TODO: how to handle sub_resolution
            ipp,
        ),
        # method="Nelder-Mead",
        method="BFGS",
    )

    return r_vec, result.x[0], result.x[1], result.fun


def optimize_grid_gmf_np(
    tx: npt.NDArray[np.complexfloating],
    ipp: npt.NDArray,
    exp_params: ExpParams,
    cfg_params: OptimizeCfgParams,
    pro_params: OptimizeProParams,
    r_vec: float,
    v_vec: float,
    a_vec: float,
) -> tuple[float, float, float, float]:
    """
    Maximize the Generalized Matched Filter GMF value using function
    optimization in continuous variable space.

    Args:
        tx: Transmitted signal
        ipp: The entire sample vector
        exp_params: Experiment parameters
        cfg_params: Configuration parameters
        pro_params: Process parameters
        gmf_start: The r/v/a value from the previous cohhind analys

    Returns
        Optimized r_vec, v_vec, a_vec, val

    """
    sample_inds = pro_params.il0_rx_window_indices

    sample_rate = exp_params.sample_rate
    wavelength = exp_params.wavelength
    min_rg = cfg_params.min_range_gate
    accel_res = pro_params.acceleration_step

    delta_a = 2 * accel_res
    res_a = 20

    max_time = exp_params.t_ipp_usec * 1e-6 * cfg_params.n_ipp
    max_velocity_change = 0.5 * (a_vec + np.sign(a_vec) * delta_a * 0.5) * max_time**2

    delta_v = 2 * max_velocity_change
    res_v = 20
    v_mat, a_mat = np.meshgrid(
        np.linspace(v_vec - delta_v * 0.5, v_vec + delta_v * 0.5, num=res_v),
        np.linspace(a_vec - delta_a * 0.5, a_vec + delta_a * 0.5, num=res_a),
    )

    rg0 = np.floor((r_vec / constants.c) * sample_rate).astype(np.int64)
    rel_rg0 = rg0 - min_rg - 1
    inds = sample_inds + rel_rg0
    sample_t = inds / sample_rate
    rx = ipp[inds]
    gmf_mat = np.zeros_like(v_mat)

    for ind in range(res_v):
        r = (
            r_vec
            + v_mat[:, ind, None] * sample_t[None, :]
            + 0.5 * a_mat[:, ind, None] * sample_t[None, :] ** 2.0
        )
        phase = 2.0 * np.pi * np.mod(r / wavelength, 1)
        model_signal = tx[None, :] * np.exp(-1j * phase)  # TODO: How to handle the sub_res

        decoded_echo = rx[None, :] * model_signal

        gmf_mat[:, ind] = np.abs(np.sum(decoded_echo, axis=1)) ** 2

    select = np.arange(res_v)
    a_inds = np.argmax(gmf_mat, axis=0)
    v_ind = np.argmax(gmf_mat[a_inds, select])
    a_ind = a_inds[v_ind]
    fun = gmf_mat[a_ind, v_ind]

    return r_vec, v_mat[a_ind, v_ind], a_mat[a_ind, v_ind], fun
