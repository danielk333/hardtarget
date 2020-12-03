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

comm = MPI.COMM_WORLD

def analyze_gmf(conf):

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
        
    n_ints=int(n.floor((b[1]-b[0])/(conf.ipp*conf.n_ipp)))/conf.n_cohints_per_file

    # parallel
    for ni in range(comm.rank,n_ints,comm.size):
        print(conf.ipp)
        print(conf.n_cohints_per_file)
        print(b[0])
        print(ni)
        fi=ni*conf.ipp*conf.n_ipp*conf.n_cohints_per_file + b[0]
        
        print("rank %d %s"%(comm.rank,stuffr.unix2datestr(fi/conf.sample_rate)))
        hdname="%s/%s"%(conf.output_dir,stuffr.sec2dirname(fi/conf.sample_rate))
        fname="%s/gmf-%d.h5"%(hdname,fi)

        gmf_max=n.zeros([conf.n_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_dc=n.zeros([conf.n_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_v=n.zeros([conf.n_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_a=n.zeros([conf.n_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_txp=n.zeros(conf.n_cohints_per_file,dtype=n.float32)
        
        if os.path.exists(fname):
            print("skipping %d, file already exists"%(fi))
        else:
            for i in range(conf.n_cohints_per_file):
                i0=fi + i*conf.ipp*conf.n_ipp

                gmf_max[i,:],gmf_dc[i,:],gmf_v[i,:],gmf_a[i,:],gmf_txp[i] = g.analyze_ipps(d,i0,conf)

            os.system("mkdir -p %s"%(hdname))
            print("writing %s"%(fname))
            ho=h5py.File(fname,"w")
            ho["gmf"]=gmf_max
            ho["gmf_dc"]=gmf_dc
            ho["a"]=gmf_a
            ho["v"]=gmf_v
            ho["tx_pwr"]=gmf_txp
            ho["i0"]=i0
            ho.close()
                

if __name__ == "__main__":
    if len(sys.argv) == 2:
        conf=go.gmf_opts(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
    print(conf)
    analyze_gmf(conf)
