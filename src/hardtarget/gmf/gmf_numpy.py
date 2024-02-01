import numpy as np
import scipy.fftpack as fft


def gmfnp(z_tx, z_rx, gmf_variables, gmf_params):
    """
    Compute the output of the Generalized Matched Filter GMF
    # TODO: Clean up these XX's and the docstring

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

    # number of range gates is input from user
    n_acc = acc_phasors.shape[0]
    zero_acc_ind = np.argmin(gmf_params["DER"]["accelerations"])
    zero_freq_ind = np.argmin(gmf_params["DER"]["fvec"])

    for ri, rg in enumerate(rgs):
        zr = z_rx[rx_window_indices + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * z_tx).reshape(-1, frequency_decimation), axis=-1)
        decimated_signal = np.zeros((gmf_params["DER"]["dec_signal_length"], ), dtype=np.complex64)

        for ai in range(n_acc):
            decimated_signal[dec_rx_window_indices] = acc_phasors[ai] * echo
            _gmfo = np.abs(fft.fft(decimated_signal)) ** 2
            mi = np.argmax(_gmfo)
            if ai == zero_acc_ind:
                # gmf_dc_vec is the range-dependent noise floor
                gmf_variables.dc[ri] = _gmfo[zero_freq_ind]

            if _gmfo[mi] > gmf_variables.vals[ri]:
                gmf_variables.vals[ri] = _gmfo[mi]
                # index of doppler that gives highest integrated energy at this range gate
                gmf_variables.v_ind[ri] = mi
                # index of acceleration that gives highest integrated energy at this range gate
                gmf_variables.a_ind[ri] = ai
    # Finished!


def gmfnp_no_reduce(z_tx, z_rx, gmf_variables, gmf_params):
    acc_phasors = gmf_params["DER"]["acceleration_phasors"]
    rgs = gmf_params["DER"]["rgs"]
    frequency_decimation = gmf_params["PRO"]["frequency_decimation"]
    rx_window_indices = gmf_params["DER"]["rx_window_indices"]
    dec_rx_window_indices = gmf_params["DER"]["dec_rx_window_indices"]

    # number of range gates is input from user
    n_acc = acc_phasors.shape[0]
    zero_acc_ind = np.argmin(gmf_params["DER"]["accelerations"])
    zero_freq_ind = np.argmin(gmf_params["DER"]["fvec"])

    for ri, rg in enumerate(rgs):
        zr = z_rx[rx_window_indices + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * z_tx).reshape(-1, frequency_decimation), axis=-1)
        decimated_signal = np.zeros((gmf_params["DER"]["dec_signal_length"], ), dtype=np.complex64)

        for ai in range(n_acc):
            decimated_signal[dec_rx_window_indices] = acc_phasors[ai] * echo
            _gmfo = np.abs(fft.fft(decimated_signal)) ** 2
            if ai == zero_acc_ind:
                # gmf_dc_vec is the range-dependent noise floor
                gmf_variables.dc[ri] = _gmfo[zero_freq_ind]

            gmf_variables.vals[ri, :, ai] = _gmfo
