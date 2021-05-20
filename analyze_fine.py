#!/usr/bin/env python

from mpi4py import MPI
import h5py
import numpy as n
import os
import time
import sys

import digital_rf as drf
import gmf_opts as go
import stuffr
import gmf as g
import gmf_cpu_numpy as gcpu
import glob
import matplotlib.pyplot as plt
import scipy.constants as c
import analyze_gmf as ag

comm = MPI.COMM_WORLD


def run_cohint(dirname="/media/j/ebd77b41-7efd-4238-b6f8-2b17bc33c84c/debsim",
               sr=1e6,
               r0=1000e3,
               tx_len=2000,
               ipp=10000,
               dr=100,   # gates
               n_ipp=10):
    cfg ="""
    [config]
    
    ground_clutter_length=0
    min_acceleration=0.0
    max_acceleration=200.0
    acceleration_resolution=0.01
    save_parameters=true
    doppler_sign=1.0
    rx_channel="ch000"
    tx_channel="tx"
    radar_frequency=230e6
    output_dir="./spade_fine"
    debug_plot=false
    debug_plot_acc=false
    debug_print=false
    round_trip_range=false
    reanalyze=true
    num_cohints_per_file=1
    use_gpu=false
    snr_thresh=10.0
    range_gate_step=1
"""

    ri0 = int((sr*r0/c.c))-dr/2
    
    with open("cfg/fine.ini","w") as f:
        f.writelines(cfg)
        # sample-rate specific configuration options
        f.write("n_range_gates=%d\n"%(dr))
        f.write("data_dirs=[\"%s\"]\n"%(dirname))
        f.write("ipp=%d\n"%(ipp))
        f.write("tx_pulse_length=%d\n"%(tx_len))
        f.write("sample_rate=%d\n"%(int(sr)))
        f.write("range_gate_0=%d\n"%(ri0))
        f.write("frequency_decimation=%d\n"%(25))
        f.write("n_ipp=%d\n"%(n_ipp))
    
    conf=go.gmf_opts("cfg/fine.ini")
    # remove old analysis
    os.system("rm spade_fine/*/*.h5")
    ag.analyze_gmf(conf,n_ints=1)

    fl=glob.glob("spade_fine/*/gmf*.h5")
    fl.sort()
    h=h5py.File(fl[0],"r")
    gmf=h["gmf"][()]
    v=h["v"][()]
    a=h["a"][()]
    print("gmf len %d"%(len(gmf[0,:])))
    mi=n.argmax(gmf[0,:])
    rgi = ri0+mi
    r00 = c.c*rgi/sr
    v00 = v[0,mi]
    a00 = a[0,mi]    
    h.close()
    return(r00,v00,a00)


def fine_tune(conf,
              noise_pwr=1.0):
    
    d=drf.DigitalRFReader(conf.data_dirs)
    b_rx=d.get_bounds(conf.rx_channel)
    b_tx=d.get_bounds(conf.tx_channel)
    
    print("RX bounds %d-%d TX bounds %d-%d"%(b_rx[0],b_rx[1],b_tx[0],b_tx[1]))
    print("RX bounds %s-%s TX bounds %s-%s"%(stuffr.unix2datestr(b_rx[0]/conf.sample_rate),
                                             stuffr.unix2datestr(b_rx[1]/conf.sample_rate),
                                             stuffr.unix2datestr(b_tx[0]/conf.sample_rate),
                                             stuffr.unix2datestr(b_tx[1]/conf.sample_rate)))
    
    print("Number of parallel processes %d"%(comm.size))
    
    if conf.t0 == None or conf.t1 == None:
        b=b_rx
    else:
        b=(int(t0*conf.sample_rate),int(t1*conf.sample_rate))

    fl=glob.glob("%s/*/gmf*.h5"%(conf.output_dir))
    fl.sort()
    
    print("Found %d GMF outputs"%(len(fl)))
    res=[]

    for f in fl:
        print(f)
        h=h5py.File(f,"r")

        gmf=h["gmf"][()]
        v=h["v"][()]
        gmf_dc=h["gmf_dc"][()]
        i0 = h["i0"][()]
        print(i0)
        ipp=conf.ipp
        n_ipp=conf.n_ipp
        for i in range(conf.num_cohints_per_file):
            mi=n.argmax(gmf[i,:])
            rgi = conf.range_gate_0+conf.range_gate_step*mi
            r0 = c.c*rgi/conf.sample_rate
            v0 = v[i,mi]
            snr_db=10.0*n.log10((gmf[i,mi]-n.sqrt(noise_pwr))**2.0/noise_pwr)

            r00,v00,a00=run_cohint(dirname="/media/j/ebd77b41-7efd-4238-b6f8-2b17bc33c84c/debsim",
                                   sr=conf.sample_rate,
                                   r0=r0,
                                   tx_len=conf.tx_pulse_length,
                                   ipp=conf.ipp,
                                   dr=100,   # gates
                                   n_ipp=n_ipp)

            
            print("Found range %1.5f,%1.5f (km) vel %1.5f,%1.5f (km/s) a %1.5f snr %1.2f (dB)"%(r0/1e3,r00/1e3,v0/1e3,v00/1e3,a00,snr_db))
            xhat=gcpu.analyze_ipps_fine(d,conf,i0+i*ipp*n_ipp,r0=r00,v0=v00,a0=a00,plott=False,noise_pwr=noise_pwr)
            res.append(xhat)
            
        h.close()
    return(xhat)
    
    

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(sys.argv[1])
        conf=go.gmf_opts(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
    print(conf)
    fine_tune(conf)
