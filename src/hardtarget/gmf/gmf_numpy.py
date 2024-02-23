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
    dec_read_length = gmf_params["PRO"]["decimated_read_length"]

    # number of range gates is input from user
    n_acc = acc_phasors.shape[0]
    for ri, rg in enumerate(rel_rgs):
        drg = int(rg // frequency_decimation)
        zr = z_rx[il1_rx_win + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * z_tx).reshape(-1, frequency_decimation), axis=-1)
        dec_signal = np.zeros((dec_read_length,), dtype=np.complex64)

        for ai in range(n_acc):
            dec_signal[il0_dec_rx_win + drg] = acc_phasors[ai] * echo
            ft2 = np.abs(fft.fft(dec_signal)) ** 2
            mi = np.argmax(ft2)
            if ai == 0:
                # zero-frequency (DC) component in scipy.fft.fft output[0] according to docs
                # used to get range-dependent noise floor
                gmf_variables.dc[ri] = ft2[0]

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
        method="Nelder-Mead",
        # method="BFGS",
    )
    x = np.array([gmf_start[0], result.x[0], result.x[1]])
    return x, result.fun


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
