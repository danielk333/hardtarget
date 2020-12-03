import digital_rf as drf
import numpy as n
import h5py
import stuffr
import scipy.fftpack as fft
import time
import scipy.constants as c
import scipy.io as sio

import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank=comm.rank
size=comm.size

def analyze_ipps(d,i0,o):
    if o.debug_plot_data_read:
        import matplotlib.pyplot as plt
   
    # we can use one of several implementations
    if o.use_gpu:
        import gmfgpu as g
    elif o.use_python:
        import gmf_cpu_numpy as g
    else:
        import gmf_cpu as g
        
    cput0=time.time()
    
    # read data vector with n_ipps, and a little extra
    z=d.read_vector_c81d(i0,(o.n_ipp+o.n_extra)*o.ipp,o.rx_channel)

    if o.debug_plot_data_read:
        plt.plot(z.real)
        plt.plot(z.imag)
        plt.show()
    # make a separate copy to hold transmit pulse, and the echo
    z_rx=n.copy(z)
    
    if o.tx_channel != o.rx_channel:
        z=d.read_vector_c81d(i0,(o.n_ipp+o.n_extra)*o.ipp,o.tx_channel)
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

    if o.debug_plot_data_read:
#        import matplotlib.pyplot as plt
        plt.plot(z_tx.real)
        plt.plot(z_tx.imag)
        plt.title("tx")
        plt.show()
        plt.plot(z_rx.real)
        plt.plot(z_rx.imag)
        plt.title("rx")
        plt.show()

    # maximum match function value
    gmf_vec=n.zeros(o.n_range_gates,dtype=n.float32)
    # best fitting range-rate
    v_vec=n.zeros(o.n_range_gates,dtype=n.float32)
    # best fitting range-rate change
    a_vec=n.zeros(o.n_range_gates,dtype=n.float32)
    # 0-frequency gmf output
    gmf_dc_vec=n.zeros(o.n_range_gates,dtype=n.float32)
    
    if tx_amp > 1.0:
        g.gmf(z_tx, z_rx, o.acc_phasors, o.rgs_float, o.frequency_decimation, gmf_vec, gmf_dc_vec, v_vec, a_vec, rank)
        
    if o.debug_plot_data_read:
        plt.plot(gmf_vec)
        plt.show()
    
    mri=n.argmax(gmf_vec)
    if o.debug_gmf_output:
        print("GMF=%1.2g r_max=%1.2f (km) vel_max=%1.2f (km/s) a_max=%1.2f (m/s**2)"%(n.max(gmf_vec),o.ranges[mri],o.range_rates[int(v_vec[mri])]/1e3,o.accs[int(a_vec[mri])]))
    
    cput1=time.time()
    if o.debug_gmf_output:
        print("time %1.2f cpu/real %1.2f"%(cput1-cput0,
                                           (cput1-cput0)/(o.n_ipp*o.ipp/o.sample_rate)))

    avec=o.accs[n.array(a_vec,dtype=n.int)]
    vvec=o.range_rates[n.array(v_vec,dtype=n.int)]
    
    return(gmf_vec,gmf_dc_vec,vvec,avec,tx_amp**2.0)
