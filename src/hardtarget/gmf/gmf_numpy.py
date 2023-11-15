import numpy as np
import scipy.fftpack as fft


def gmf_numpy(z_tx, z_rx, a_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec, rx_window_indecies):
    """
    Compute the output of the Generalized Matched Filter GMF

    Parameters:
    z_tx: complex vector [n_fft] of transmitter samples
    z_rx: complex vector [n_fft] of receiver samples
    a_phasors: array [n_acc, n_fft//dec]  of phase for acceleration grid

    Here, n_fft is the length of the coherent integration in samples

    rgs: integer vector [n_rngs], start index of each range gate
    dec: integer, boxcar decimation factor to apply after accel matching

    Output parameters:
        gmf_vec: real vector [n_rngs] max value of gmf (power) across vel/acc
        gmf_dc_vec: real vector [n_rngs] output of gmf (power) at zero frequency
        a_vec: integer vector [n_rngs] index of a_phasors that produced max output at each range
        v_vec: integer vector [n_rngs] index of velocity that produced max output at each range

    """

    # number of range gates is input from user
    n_acc = a_phasors.shape[0]

    for ri, rg in enumerate(rgs):
        zr = z_rx[rx_window_indecies + rg]
        # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter), decimate
        echo = np.sum((zr * z_tx).reshape(-1, dec), axis=-1)

        # for ai, a in enumerate (accels):
        for ai in range(n_acc):
            _gmfo = np.abs(fft.fft(a_phasors[ai] * echo, len(echo))) ** 2
            mi = np.argmax(_gmfo)
            if ai == 0:
                # gmf_dc_vec is the range-dependent noise floor
                gmf_dc_vec[ri] = _gmfo[0]

            if _gmfo[mi] > gmf_vec[ri]:
                gmf_vec[ri] = _gmfo[mi]
                # index of doppler that gives highest integrated energy at this range gate
                v_vec[ri] = mi
                # index of acceleration that gives highest integrated energy at this range gate
                a_vec[ri] = ai
    # Finished!
