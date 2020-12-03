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


import digital_rf as drf
import matplotlib.pyplot as plt
import numpy as n
import h5py
import stuffr
import scipy.fftpack as fft
import time
import scipy.constants as c
#import pyfftw

import os


def analyze_ipps(d,i0,o,mode=0,plott=False):
    
    # read data vector with n_ipps, and a little extra
    z=d.read_vector_c81d(i0,(o.n_ipp+o.n_extra)*o.ipp,o.ch)

    if plott:
        # plot echo
        plt.plot(z.real)
        plt.plot(z.imag)
        plt.title("Echo")
        plt.show()

    # make a separate copy to hold transmit pulse, and the echo
    z_tx=n.copy(z)
    z_rx=n.copy(z)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_tx=z_tx*o.tx_stencil
    z_rx=z_rx*o.rx_stencil

    # truncate the tx vector to be exactly the length of n_ipp
    z_tx=z_tx[0:(o.n_ipp*o.ipp)]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    z_tx=n.conj(z_tx)

    if plott:
        # plot tx
        plt.plot(z_tx.real)
        plt.plot(z_tx.imag)
        plt.title("Only tx")
        plt.show()
        
        # plot echo
        plt.plot(z_rx.real)
        plt.plot(z_rx.imag)
        plt.title("Only echo")
        plt.show()

    cput0=time.time()
    max_r=0
    max_v=0
    max_a=0
    max_mf=0
    gmf_vec=n.zeros(o.n_r,dtype=n.float32)
    v_vec=n.zeros(o.n_r,dtype=n.float32)
    a_vec=n.zeros(o.n_r,dtype=n.float32)        
    gmf_dc_vec=n.zeros(o.n_r,dtype=n.float32)

    # implement this in C
    #
    # gmf.mf(z_tx, o.n_fft, z_rx,len(z_rx), o.acc_phasors, o.n_accs, o.rgs,o.fdec,gmf_vec,gmf_dc_vec,v_vec,a_vec)
    for ri,rg in enumerate(o.rgs):
        # range matching echo*conj(tx)
        echo=stuffr.decimate(z_rx[rg:(rg+o.n_fft)]*z_tx,dec=o.fdec)
        
        # go through all accelerations
        for ai,a in enumerate(o.accs):
            # go through all doppler shifts with FFT (this is a grid search of
            # all possible doppler velocities)
            gmf=n.abs(fft.fft(o.acc_phasors[ai,:]*echo))
            mi=n.argmax(gmf)
            
            if ai==0:
                gmf_dc_vec[ri]=gmf[0]
            
            if gmf[mi]>gmf_vec[ri]:
                gmf_vec[ri]=gmf[mi]
                v_vec[ri]=o.vel_vec[mi]
                a_vec[ri]=a


    if plott:
        plt.plot(gmf_vec)
        plt.show()
    cput1=time.time()
    print("time %1.2f cpu/real %1.2f"%(cput1-cput0,(cput1-cput0)/(o.n_ipp*o.ipp/o.sr)))
    mri=n.argmax(gmf_vec)
    print("GMF=%1.2g r_max=%1.2f (km) vel_max=%1.2f (km/s) a_max=%1.2f (m/s**2)"%(o.fdec*n.max(gmf_vec),o.range_vec[mri],v_vec[mri]/1e3,a_vec[mri]))

    hdname="%s/%s"%(o.output_dname,stuffr.sec2dirname(i0/o.sr))
    os.system("mkdir -p %s"%(hdname))
    fname="%s/gmf-%d-%d.h5"%(hdname,mode,i0)
    print("writing %s"%(fname))
    ho=h5py.File(fname,"w")
    ho["gmf"]=gmf_vec
    ho["gmf_dc"]=gmf_dc_vec    
    ho["a"]=a_vec
    ho["v"]=v_vec
    ho["rg0"]=o.rg0
    ho["mode"]=mode
    ho["i0"]=i0
    ho.close()

