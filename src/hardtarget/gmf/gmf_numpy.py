import numpy as np
import scipy.fft as fft
import scipy.constants as constants
import scipy.optimize as sco


def gmfnp(z_tx, z_rx, gmf_variables, gmf_params):
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

    acc_phasors = gmf_params["DER"]["acceleration_phasors"]
    rgs = gmf_params["DER"]["rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    rx_window_indices = gmf_params["DER"]["rx_window_indices"]
    dec_rx_window_indices = gmf_params["DER"]["dec_rx_window_indices"]
    dec_signal_len = gmf_params["DER"]["dec_signal_length"]

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
            mi = np.argmax(_gmfo)
            if ai == 0:
                # zero-frequency (DC) component in scipy.fft.fft output[0] according to docs
                # used to get range-dependent noise floor
                gmf_variables.dc[ri] = _gmfo[0]

            if _gmfo[mi] > gmf_variables.vals[ri]:
                gmf_variables.vals[ri] = _gmfo[mi]
                # index of doppler that gives highest integrated energy at this range gate
                gmf_variables.v_ind[ri] = mi
                # index of acceleration that gives highest integrated energy at this range gate
                gmf_variables.a_ind[ri] = ai
    # Finished!


def gmfnp_optimize(z_tx, z_ipp, gmf_params, gmf_start):
    """Maximize the Generalized Matched Filter GMF value using function
    optimization in continuous variable space.

    Here z_ipp is the entire rx sample vector, not the stenciled one as the
    stencils are done by the gmf forward model

    # TODO: make this function stable and working
    """
    def neg_gmf_direct(x, sample_t, wavelength, sample_rate, z_tx, z_ipp):
        r = x[0] + x[1]*sample_t + 0.5*x[2]*sample_t**2.0
        t = r / constants.c
        phase = np.exp(-1j*2.0*np.pi*r/wavelength)
        model_signal = phase * z_tx

        samples = np.round(sample_rate * t).astype(np.int64)
        decoded_echo = z_ipp[samples] * model_signal

        return -np.abs(np.sum(decoded_echo))

    # TODO: make so that the input parameters for minimize can be customized trough the config file
    # such as optimization limits and method
    result = sco.minimize(
        neg_gmf_direct,
        gmf_start,
        args=(
            gmf_params["DER"]["sample_times"],
            gmf_params["EXP"]["wavelength"],
            gmf_params["EXP"]["sample_rate"],
            z_tx,
            z_ipp,
        ),
        method="Nelder-Mead",
    )
    return result


def gmfnp_no_reduce(z_tx, z_rx, gmf_variables, gmf_params):
    acc_phasors = gmf_params["DER"]["acceleration_phasors"]
    rgs = gmf_params["DER"]["rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    rx_window_indices = gmf_params["DER"]["rx_window_indices"]
    dec_rx_window_indices = gmf_params["DER"]["dec_rx_window_indices"]
    dec_signal_len = gmf_params["DER"]["dec_signal_length"]

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
            indexing = tuple(
                z
                for zi, z in enumerate([ri, 0, ai])
                if not gmf_params["PRO"]["reduce_axis"][zi]
            )
            if gmf_params["PRO"]["reduce_axis"][1]:
                vi = np.argmax(_gmfo)
                current_val = _gmfo[vi]
                old_val = gmf_variables.vals[*indexing]
                if current_val > old_val:
                    gmf_variables.vals[*indexing] = current_val
                    gmf_variables.r_ind[*indexing] = ri
                    gmf_variables.v_ind[*indexing] = vi
                    gmf_variables.a_ind[*indexing] = ai
            else:
                for vi in range(len(_gmfo)):
                    current_val = _gmfo[vi]
                    old_val = gmf_variables.vals[*indexing]
                    if current_val > old_val:
                        gmf_variables.vals[*indexing] = current_val
                        gmf_variables.r_ind[*indexing] = ri
                        gmf_variables.v_ind[*indexing] = vi
                        gmf_variables.a_ind[*indexing] = ai
