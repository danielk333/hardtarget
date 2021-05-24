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


def run_cohint(d,conf,i0,r0):
    """
    run gmf with a finer resolution. reset range gate setting with existing high res configuration
    """
    n_rg=conf.n_range_gates
    ri0 = int(n.round((r0/c.c)*conf.sample_rate)) - int(n_rg/2)
    conf.set_n_ranges(ri0,n_rg)
    
    gmf_max, gmf_dc, gmf_v, gmf_a, gmf_txp = g.analyze_ipps(d,i0,conf)

    mi=n.argmax(gmf_max)
    gmf = gmf_max[mi]
    rgi = ri0+mi
    r00 = c.c*rgi/conf.sample_rate
    v00 = gmf_v[mi]
    a00 = gmf_a[mi]
    
    return(gmf,r00,v00,a00)

def fine_tune(conf,
              conf_fine,
              noise_pwr=1.0):
    """
    fine tuned analysis
    """

    print(conf.data_dirs)
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

            # refine guess
            gmf00,r00,v00,a00=run_cohint(d,
                                         conf_fine,
                                         i0,
                                         r0)


            # an even better but slower alternative is the following:
            # gmf_cpu_numpy.analyze_ipps_fine
                                  
            best_snr=10.0*n.log10((gmf00-n.sqrt(noise_pwr))**2.0/noise_pwr)
            
            print("Found range %1.5f,%1.5f (km) vel %1.5f,%1.5f (km/s) a %1.5f snr %1.2f,%1.2f (dB)"%(r0/1e3,r00/1e3,v0/1e3,v00/1e3,a00,snr_db,best_snr))
            xhat=n.array([best_snr,r00,v00,a00])
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
