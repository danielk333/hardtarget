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
#import matplotlib.pyplot as plt
import numpy as n
import h5py
import stuffr
import scipy.fftpack as fft
import time
import scipy.constants as c
import scipy.io as sio
#import gmf
#import gmfgpu as gmf

#import pyfftw


import os

def analyze_ipps(d,i0,o,mode=0,rank=0,plott=False,fname="test.h5",quiet=False):
    if o.use_gpu:
        import gmfgpu as gmf
    else:
        import gmf
        
    cput0=time.time()    
    # read data vector with n_ipps, and a little extra
    z=d.read_vector_c81d(i0,(o.n_ipp+o.n_extra)*o.ipp,o.ch)

    # make a separate copy to hold transmit pulse, and the echo
    z_rx=n.copy(z)
    if o.tx_ch != o.ch:
        z=d.read_vector_c81d(i0,(o.n_ipp+o.n_extra)*o.ipp,o.tx_ch)
    z_tx=n.copy(z)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_tx=z_tx*o.tx_stencil
    z_rx=z_rx*o.rx_stencil

    # truncate the tx vector to be exactly the length of n_ipp
    z_tx=z_tx[0:(o.n_ipp*o.ipp)]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_amp=n.sqrt(n.sum(n.abs(z_tx)**2.0))
    z_tx=n.conj(z_tx)/tx_amp

    if plott:
        import matplotlib.pyplot as plt
        plt.plot(z_tx.real)
        plt.plot(z_tx.imag)    
        plt.show()
        plt.plot(z_rx.real)
        plt.plot(z_rx.imag)    
        plt.show()
    
    gmf_vec=n.zeros(o.n_r,dtype=n.float32)
    v_vec=n.zeros(o.n_r,dtype=n.float32)
    a_vec=n.zeros(o.n_r,dtype=n.float32)        
    gmf_dc_vec=n.zeros(o.n_r,dtype=n.float32)
    if tx_amp > 1.0:
        gmf.gmf(z_tx, z_rx, o.acc_phasors, o.rgs_float, o.fdec, gmf_vec, gmf_dc_vec, v_vec, a_vec, rank)

    
    mri=n.argmax(gmf_vec)
    if not quiet:
        print("GMF=%1.2g r_max=%1.2f (km) vel_max=%1.2f (km/s) a_max=%1.2f (m/s**2)"%(o.fdec*n.max(gmf_vec),o.range_vec[mri],o.vel_vec[int(v_vec[mri])]/1e3,o.accs[int(a_vec[mri])]))
    



    if o.output_mode == "old":
        hdname="%s/%s"%(o.output_dname,stuffr.sec2dirname(i0/o.sr))
        os.system("mkdir -p %s"%(hdname))
        fname="%s/gmf-%d-%d.h5"%(hdname,mode,i0)
        print("writing %s"%(fname))
        ho=h5py.File(fname,"w")
        ho["gmf"]=n.sqrt(gmf_vec)
        ho["gmf_dc"]=n.sqrt(gmf_dc_vec)
        ho["a"]=o.accs[n.array(a_vec,dtype=n.int)]
        ho["v"]=o.vel_vec[n.array(v_vec,dtype=n.int)]
        ho["tx_pwr"]=tx_amp**2.0
        ho["rg0"]=o.rg0
        ho["mode"]=mode
        ho["i0"]=i0
        ho.close()
    else:
        pass
#        hdname="%s/%s"%(o.output_dname,stuffr.sec2dirname(i0/o.sr))
 #       os.system("mkdir -p %s"%(hdname))
        
  #      fname="%s/gmf-%d-%d.mat"%(hdname,mode,i0)
   #     print("writing %s"%(fname))
    #    sio.savemat(fname,{"gmf":n.sqrt(gmf_vec),
     #                      "gmf_dc":n.sqrt(gmf_dc_vec),
      #                     "a":o.accs[n.array(a_vec,dtype=n.int)],
       #                    "v":o.vel_vec[n.array(v_vec,dtype=n.int)],
        #                   "mode":mode})
        
    cput1=time.time()
    if not quiet:
        print("time %1.2f cpu/real %1.2f"%(cput1-cput0,(cput1-cput0)/(o.n_ipp*o.ipp/o.sr)))

    gmfvec=n.sqrt(gmf_vec)
    avec=o.accs[n.array(a_vec,dtype=n.int)]
    vvec=o.vel_vec[n.array(v_vec,dtype=n.int)]
    
    return(gmfvec,vvec,avec)
