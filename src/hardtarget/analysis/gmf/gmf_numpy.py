#!/usr/bin/env pyth
#
# Example of range-Doppler-acceleration matched filter (Generalized Match Function)
#
# # example1
#i0=1463100914401305
#rg0=2000#4400
## example 2
##i0=1463106122981305
#rg0=5425
# example 3 (range aliased echo!)
#i0=1463107264431305
#rg0=8145+10000


import os
import time

import h5py

import numpy as np
import scipy.fftpack as fft
import scipy.constants as c
import scipy.optimize as sio

import digital_rf as drf


from hardtarget.utilities import read_vector_c81d

# g.gmf(z_tx, z_rx, o.acc_phasors, o.rgs_float, o.frequency_decimation, gmf_vec, gmf_dc_vec, v_vec, a_vec, comm.rank)
def gmf_numpy(z_tx, z_rx, a_phasors, rgs, dec, 
            gmf_vec, gmf_dc_vec, v_vec, a_vec, rank=None):
    """
    Compute the output of the Generalized Matched Filter GMF

    Parameters:
    z_tx: complex vector [n_fft] of transmitter samples
    z_rx: complex vector [n_fft + n_extra*ipp] of receiver samples
    a_phasors: array [n_acc, n_fft//dec]  of phase for acceleration grid

    rank: MPI rank (not used)

    Here, n_fft is the length of the coherent integration in samples

    rgs: integer vector [n_rngs], start index of each range gate
    dec: integer, boxcar decimation factor to apply after accel matching

    Output parameters:
        gmf_vec: real vector [n_rngs] max value of gmf (power) across vel/acc
        gmf_dc_vec: real vector [shape??] output of gmf (power) at zero frequency
        a_vec: integer vector [n_rngs] index of a_phasors that produced max output at each range
        v_vec: integer vector [n_rngs] index of velocity that produced max output at each range
    """

    # TODO:
    # defaults for acc_phasors
    # defaults for rgs
    # defaults for dec


    # Stencil and CC tx waveform here, or on the outside?
    # => on the outside

    # n_range_gates = (len(z_rx) -len(z_tx)) // dec    # ??
    # number of range gates is input from user
    n_acc = a_phasors.shape[0]

    # misnamed parameter: n_fft is length of coherent integration
    n_fft = len(z_tx)

    # number of frequency bins is length of coherent integration after boxcar decimation
    n_vel = n_fft//dec

    # GA = np.zeros((n_acc, n_range_gates))
    # GV = np.zeros((n_vel, n_range_gates))

    for ri, rg in enumerate(rgs):

        rg_idx = rg.astype(np.int32)
        zr = z_rx[rg_idx:(rg_idx+n_fft)]
        # echo = stuffr.decimate(zr * z_tx, dec=dec)     # Matched filter output, stacked IPPs, bandwidth-reduced (boxcar filter)
        echo = np.sum((zr * z_tx).reshape(-1, dec), axis=-1)

        # for ai, a in enumerate (accels):
        for ai in range(n_acc):
            _gmfo = np.abs(fft.fft(a_phasors[ai] * echo, len(echo)))**2
            mi = np.argmax(_gmfo)
            # GA[ai, ri] = _gmfo[mi]

            if ai == 0:
                gmf_dc_vec[ri] = _gmfo[0]       # gmf_dc_vec is the range-dependent noise floor

            if _gmfo[mi] > gmf_vec[ri]:
                gmf_vec[ri] = _gmfo[mi]
                v_vec[ri]   = mi        # index of doppler that gives highest integrated energy at this range gate
                a_vec[ri]   = ai        # index of acceleration that gives highest integrated energy at this range gate

    # return gmf_vec, gmf_dc_vec, a_vec, v_vec
    # Finished!


