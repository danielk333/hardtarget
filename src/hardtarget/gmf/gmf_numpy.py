import numpy as np
import scipy.fft as fft
import scipy.constants as constants
import scipy.optimize as sco


def fast_gmf_np(z_tx, z_rx, gmf_variables, gmf_params):
    """
    Compute the output of the Generalized Matched Filter GMF
    # TODO: Clean up these XX's and the docstring

    # TODO: should rx sliding window change based on acceleration and velocity?
    # technically the signal can start shifting samples at the end

    Parameters:
    z_tx: complex vector [XX] of transmitter samples
    z_rx: complex vector [XX] of receiver samples
    a_phasors: array [n_acc, XX//dec]  of phase for acceleration grid

    Here, n_fft is the length of the coherent integration in samples

    rgs: integer vector [XX], start index of each range gate
    dec: integer, boxcar decimation factor to apply after accel matching

    Output parameters:
        gmf_vec: real vector [XX] max value of gmf (power) across vel/acc
        gmf_dc_vec: real vector [XX] output of gmf (power) at zero frequency
        a_vec: integer vector [XX] index of a_phasors that produced max output at each range
        v_vec: integer vector [XX] index of velocity that produced max output at each range

    """

    acc_phasors = gmf_params["DER"]["fgmf_acceleration_phasors"]
    acc_inds = gmf_params["DER"]["inds_accelerations"]
    rel_rgs = gmf_params["DER"]["rel_rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    il1_rx_win = gmf_params["DER"]["il1_rx_window_indices"]
    il0_dec_rx_win = gmf_params["DER"]["il0_dec_rx_window_indices"]
    dec_signal_len = gmf_params["PRO"]["decimated_read_length"]

    # number of range gates is input from user
    n_acc = acc_phasors.shape[0]
    for ri, rg in enumerate(rel_rgs):
        drg = int(rg // frequency_decimation)
        zr = z_rx[il1_rx_win + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * z_tx).reshape(-1, frequency_decimation), axis=-1)
        dec_signal = np.zeros((dec_signal_len,), dtype=np.complex64)
        # zero-frequency (DC) is used to get range-dependent noise floor
        gmf_variables.dc[ri] = np.abs(np.sum(echo)) ** 2

        for ai in range(n_acc):
            dec_signal[il0_dec_rx_win + drg] = acc_phasors[ai] * echo
            ft2 = np.abs(fft.fft(dec_signal)) ** 2
            mi = np.argmax(ft2)

            if ft2[mi] > gmf_variables.vals[ri]:
                gmf_variables.vals[ri] = ft2[mi]
                # index of doppler that gives highest integrated energy at this range gate
                gmf_variables.v_ind[ri] = mi
                # index of acceleration that gives highest integrated energy at this range gate
                gmf_variables.a_ind[ri] = acc_inds[ai]


def fast_dpt_np(z_tx, z_rx, gmf_variables, gmf_params):
    """Development version of GMF using discrete ambiguity function spectrum to speed up acceleration search
    """
    acc_phasors = gmf_params["DER"]["acceleration_phasors"]
    rel_rgs = gmf_params["DER"]["rel_rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    il1_rx_win = gmf_params["DER"]["il1_rx_window_indices"]
    il0_dec_rx_win = gmf_params["DER"]["il0_dec_rx_window_indices"]
    dec_read_length = gmf_params["PRO"]["decimated_read_length"]
    dec_tau_samp = gmf_params["PRO"]["dec_tau_samp"]

    for ri, rg in enumerate(rel_rgs):
        drg = int(rg // frequency_decimation)
        zr = z_rx[il1_rx_win + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * z_tx).reshape(-1, frequency_decimation), axis=-1)
        dec_signal = np.zeros((dec_read_length,), dtype=np.complex64)
        dec_signal[il0_dec_rx_win + drg] = echo

        gmf_variables.dc[ri] = np.abs(np.sum(echo)) ** 2

        dpt2 = dec_signal[dec_tau_samp:] * np.conj(dec_signal[:-dec_tau_samp])
        dpt2_spec = np.abs(fft.fftshift(fft.fft(dpt2)))
        dspec_peak = np.argmax(dpt2_spec)

        dec_signal[il0_dec_rx_win + drg] = acc_phasors[dspec_peak] * echo
        ft2 = np.abs(fft.fft(dec_signal)) ** 2
        spec_peak = np.argmax(ft2)

        gmf_variables.vals[ri] = ft2[spec_peak]
        gmf_variables.v_ind[ri] = spec_peak
        gmf_variables.a_ind[ri] = dspec_peak


def optimize_gmf_np(z_tx, z_ipp, gmf_params, gmf_start):
    """Maximize the Generalized Matched Filter GMF value using function
    optimization in continuous variable space.

    Here z_ipp is the entire rx sample vector, not the stenciled one as the
    stencils are done by the gmf forward model

    """
    sample_inds = gmf_params["DER"]["il0_rx_window_indices"]

    def neg_gmf_direct(x, r0, sample_inds, wavelength, sample_rate, tx0_samp, z_tx, z_ipp):
        rg0 = np.floor((r0 / constants.c) * sample_rate).astype(np.int64) + tx0_samp
        inds = sample_inds + rg0

        sample_t = inds / sample_rate
        r = r0 + x[0]*sample_t + 0.5*x[1]*sample_t**2.0
        phase = 2.0*np.pi*np.mod(r/wavelength, 1)
        model_signal = z_tx * np.exp(-1j*phase)

        decoded_echo = z_ipp[inds] * model_signal

        return -np.abs(np.sum(decoded_echo)) ** 2

    # TODO: make so that the input parameters for minimize can be customized trough the config file
    # such as optimization limits and method
    result = sco.minimize(
        neg_gmf_direct,
        gmf_start[1:],
        args=(
            gmf_start[0],
            sample_inds,
            gmf_params["EXP"]["wavelength"],
            gmf_params["EXP"]["sample_rate"],
            gmf_params["EXP"]["T_tx_start_samp"],
            z_tx,
            z_ipp,
        ),
        # method="Nelder-Mead",
        method="BFGS",
    )
    x = np.array([gmf_start[0], result.x[0], result.x[1]])
    return x, result.fun


def optimize_grid_gmf_np(z_tx, z_ipp, gmf_params, gmf_start):
    """Maximize the Generalized Matched Filter GMF value using function
    optimization in continuous variable space.

    Here z_ipp is the entire rx sample vector, not the stenciled one as the
    stencils are done by the gmf forward model

    """
    sample_inds = gmf_params["DER"]["il0_rx_window_indices"]

    r0, v0, a0 = gmf_start.tolist()
    sample_rate = gmf_params["EXP"]["sample_rate"]
    wavelength = gmf_params["EXP"]["wavelength"]
    min_rg = gmf_params["PRO"]["min_range_gate"]
    accel_res = gmf_params["DER"]["acceleration_step"]

    delta_a = 2*accel_res
    res_a = 20

    max_time = gmf_params["EXP"]["ipp"]*1e-6*gmf_params["PRO"]["n_ipp"]
    max_velocity_change = 0.5*(a0 + np.sign(a0)*delta_a*0.5)*max_time**2

    delta_v = 2*max_velocity_change
    res_v = 20
    v_mat, a_mat = np.meshgrid(
        np.linspace(v0 - delta_v*0.5, v0 + delta_v*0.5, num=res_v),
        np.linspace(a0 - delta_a*0.5, a0 + delta_a*0.5, num=res_a),
    )

    rg0 = np.floor((r0 / constants.c) * sample_rate).astype(np.int64)
    rel_rg0 = rg0 - min_rg - 1
    inds = sample_inds + rel_rg0
    sample_t = inds / sample_rate
    z_rx = z_ipp[inds]
    gmf_mat = np.zeros_like(v_mat)

    for ind in range(res_v):
        r = r0 + v_mat[:, ind, None]*sample_t[None, :] + 0.5*a_mat[:, ind, None]*sample_t[None, :]**2.0
        phase = 2.0*np.pi*np.mod(r/wavelength, 1)
        model_signal = z_tx[None, :] * np.exp(-1j*phase)

        decoded_echo = z_rx[None, :] * model_signal

        gmf_mat[:, ind] = np.abs(np.sum(decoded_echo, axis=1)) ** 2

    select = np.arange(res_v)
    a_inds = np.argmax(gmf_mat, axis=0)
    v_ind = np.argmax(gmf_mat[a_inds, select])
    a_ind = a_inds[v_ind]
    x = np.array([r0, v_mat[a_ind, v_ind], a_mat[a_ind, v_ind]])
    fun = gmf_mat[a_ind, v_ind]

    return x, fun


def fast_gmf_no_reduce_np(z_tx, z_rx, gmf_variables, gmf_params):
    """Slow development version of gmf to see otherwise reduced dimensions
    """
    acc_phasors = gmf_params["DER"]["acceleration_phasors"]
    rgs = gmf_params["DER"]["rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    rx_window_indices = gmf_params["DER"]["rx_window_indices"]
    dec_rx_window_indices = gmf_params["DER"]["dec_rx_window_indices"]
    dec_signal_len = gmf_params["DER"]["dec_signal_length"]

    ra = gmf_params["PRO"]["reduce_axis"]

    # number of range gates is input from user
    n_acc = acc_phasors.shape[0]
    for ri, rg in enumerate(rgs):
        zr = z_rx[rx_window_indices + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * z_tx).reshape(-1, frequency_decimation), axis=-1)
        decimated_signal = np.zeros((dec_signal_len,), dtype=np.complex64)

        for ai in range(n_acc):
            decimated_signal[dec_rx_window_indices] = acc_phasors[ai] * echo
            _gmfo = np.abs(fft.fft(decimated_signal)) ** 2
            if ai == 0:
                # gmf_dc_vec is the range-dependent noise floor
                gmf_variables.dc[ri] = _gmfo[0]
            if ra[1]:
                vi = np.argmax(_gmfo)
                new_val = _gmfo[vi]
                if ra[0]:
                    if new_val > gmf_variables.vals[ai]:
                        gmf_variables.vals[ai] = new_val
                        gmf_variables.r_ind[ai] = ri
                        gmf_variables.v_ind[ai] = vi
                elif ra[2]:
                    if new_val > gmf_variables.vals[ri]:
                        gmf_variables.vals[ri] = new_val
                        gmf_variables.v_ind[ri] = vi
                        gmf_variables.a_ind[ri] = ai
                else:
                    raise NotImplementedError("")
            else:
                inds = np.arange(len(_gmfo))
                if ra[0] and ra[2]:
                    vals = np.stack([gmf_variables.vals[:], _gmfo[:]])
                    mi = np.argmax(vals, axis=0)
                    gmf_variables.vals[:] = vals[mi, inds]
                    sel = mi == 1
                    gmf_variables.r_ind[sel] = ri
                    gmf_variables.a_ind[sel] = ai
                elif ra[0]:
                    vals = np.stack([gmf_variables.vals[:, ai], _gmfo[:]])
                    mi = np.argmax(vals, axis=0)
                    gmf_variables.vals[:, ai] = vals[mi, inds]
                    sel = mi == 1
                    gmf_variables.r_ind[sel, ai] = ri
                else:
                    vals = np.stack([gmf_variables.vals[:, ri], _gmfo[:]])
                    mi = np.argmax(vals, axis=0)
                    gmf_variables.vals[:, ri] = vals[mi, inds]
                    sel = mi == 1
                    gmf_variables.a_ind[sel, ri] = ai
