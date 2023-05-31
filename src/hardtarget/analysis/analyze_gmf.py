#!/usr/bin/env python

import h5py
import numpy as np
import os
import sys
import datetime
import digital_rf as drf
from hardtarget.analysis import gmf_opts as go
from hardtarget.analysis import analyze_ipps as g
from hardtarget.utilities import unix2datestr, sec2dirname
import configparser
        

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1
    comm = COMM_WORLD()


####################################################################
# GMF ARGS
####################################################################

def get_gmf_args(config_file):
    """
    Converts a config file to a dictionary
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    d = {}
    if 'config' in config:
        for key, value in config['config'].items():
            # Convert values to specific types
            if key in ['n_ipp', 't0', 'sample_rate', 'n_range_gates', 'range_gate_0', 'range_gate_step',
                    'ipp', 'tx_pulse_length', 'ground_clutter_length', 'num_cohints_per_file']:
                d[key] = int(value)
            elif key in ['reanalyze', 'round_trip_range']:
                d[key] = config.getboolean('config', key)
            elif key in ['rx_channel', 'tx_channel', 'output_dir']:
                d[key] = value.strip('"')       
            elif key in ['radar_frequency', 'doppler_sign', 'min_acceleration',
                    'max_acceleration', 'acceleration_resolution']:
                d[key] = float(value)
            else:                
                pass
    return d
            

####################################################################
# TASKS
####################################################################

def get_tasks (worker_idx, n_workers, n_tasks):
    """
    returns a list of task_idx for worker_idx

    n_tasks: (int) total number of tasks
    n_workers: (int) number of workers

    worker_idx: (int) index of worker : [0, ... , n_workers-1] 
    task_idx: (int) index of task : [0, ... , n_tasks-1]

    tasks are divided equally (or close to equally) among workers
    """
    return list(range(worker_idx, n_tasks, n_workers))




def filepath(dir, fi, sample_rate):
    dt = datetime.datetime.utcfromtimestamp(fi/sample_rate)
    time_string = dt.strftime("%Y-%m-%dT%H-00-00")
    return os.path.join([
        dir,
        time_string,
        f"gmf-{fi:08d}.h5"
        ])


        #print("rank %d %s"%(comm.rank, unix2datestr(fi/sample_rate)))
        #hdname="%s/%s"%(output_dir,sec2dirname(fi/sample_rate))
        #fname="%s/


####################################################################
# ANALYSE GMF
####################################################################


def analyze_gmf(drf_data, output_dir, gmf_args,
                n_ints=0,
                reanalyze=True):
    """
    Analyze data using gmf.
        - tx_channel: transmit channel
        - rx_channel: receive channel
        - t0: 
        - sample_rate:
        - ipp: inter pulse period
        - n_ipp: number of pulses in period
        - n_range_gates: 
        - num_cohints_per_file: 
        - output_dir: directory path for results
        - reanalyze: if True - clear previous results 
        - n_ints: number of integrations?
    """
    tx_channel = gmf_args["tx_channel"]
    rx_channel = gmf_args["rx_channel"]
    t0 = gmf_args["t0"]
    sample_rate = gmf_args["sample_rate"]
    ipp = gmf_args["ipp"]
    n_ipp = gmf_args["n_ipp"]
    n_range_gates = gmf_args["n_range_gates"]
    num_cohints_per_file = gmf_args["num_cohints_per_file"]

    # clear result if already exists
    #if reanalyze:
    #    os.system("rm -f %s/*/*.h5"%(output_dir))

    # RX and TX bounds
    bounds_rx = drf_data.get_bounds(rx_channel)
    bounds_tx = drf_data.get_bounds(tx_channel)
    
    #print("RX bounds %d-%d TX bounds %d-%d"%(b_rx[0],b_rx[1],b_tx[0],b_tx[1]))
    #print("RX bounds %s-%s TX bounds %s-%s"%(unix2datestr(b_rx[0]/conf.sample_rate),
    #                                         unix2datestr(b_rx[1]/conf.sample_rate),
    #                                         unix2datestr(b_tx[0]/conf.sample_rate),
    #                                         unix2datestr(b_tx[1]/conf.sample_rate)))
    
    print("Number of parallel processes %d"%(comm.size))


    bounds = bounds_rx
    bounds = [bounds_rx[0],bounds_rx[1]]
    if t0 != None:
        bounds[0] = int(t0*sample_rate)
        

    # adjust n_ints
    if n_ints == 0:
        n_ints=int(np.floor((bounds[1]-bounds[0])/(ipp*n_ipp))/num_cohints_per_file)

    worker_idx = comm.rank
    n_workers = comm.size

    # process
    for task_idx in get_tasks(worker_idx, n_workers, n_ints):
        fi=task_idx*ipp*n_ipp*num_cohints_per_file + bounds[0]        
        #print("rank %d %s"%(comm.rank, unix2datestr(fi/sample_rate)))
        hdname="%s/%s"%(output_dir,sec2dirname(fi/sample_rate))
        fname="%s/gmf-%08d.h5"%(hdname,fi)

        gmf_max=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_dc=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_v=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_a=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
        gmf_txp=np.zeros(num_cohints_per_file, dtype=np.float32)
    
        if os.path.exists(fname):
            print("skipping %d, file already exists"%(fi))
        else:
            for i in range(num_cohints_per_file):
                i0=fi + i*ipp*n_ipp
                
                #gmf_max[i,:], gmf_dc[i,:], gmf_v[i,:], gmf_a[i,:], gmf_txp[i] = g.analyze_ipps(drf_data,i0,conf)
                #rgi=np.argmax(gmf_max[i,:])

            path = filepath(output_dir, fi, sample_rate)

            os.system("mkdir -p %s"%(hdname))
            print("writing %s"%(fname))
            
            """ ho=h5py.File(fname,"w")
            ho["gmf"]=gmf_max
            ho["gmf_dc"]=gmf_dc
            ho["a"]=gmf_a
            ho["v"]=gmf_v
            ho["tx_pwr"]=gmf_txp
            ho["i0"]=i0
            ho.close() """

if __name__ == "__main__":
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        print("config file", config_file)

        HOME = os.path.expanduser("~")
        SRCDIR = os.path.join(HOME, "Data/hard_target/leo_bpark_2.1u_NO@uhf/drf/")
        DSTDIR = "."
        drf_data = drf.DigitalRFReader(SRCDIR)
        gmf_args = get_gmf_args(config_file)    
        analyze_gmf(drf_data, DSTDIR, gmf_args)

        # conf=go.gmf_opts(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
    # print(conf)
    #analyze_gmf(conf)
