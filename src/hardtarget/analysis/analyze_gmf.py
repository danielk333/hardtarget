#!/usr/bin/env python
import argparse
import logging
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
        
LOGGER_NAME = "analyse_gmf"


####################################################################
# UTIL
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
            

def get_tasks (worker, n_tasks):
    """
    returns a list of task_idx for worker_idx

    n_tasks: (int) total number of tasks
    n_workers: (int) number of workers

    worker_idx: (int) index of worker : [0, ... , n_workers-1] 
    task_idx: (int) index of task : [0, ... , n_tasks-1]

    tasks are divided equally (or close to equally) among workers
    """
    return list(range(worker["idx"], n_tasks, worker["N"]))



def get_filepath(dir, file_idx, sample_rate):
    """
    create a file path for output
    """
    dt = datetime.datetime.utcfromtimestamp(file_idx/sample_rate)
    time_string = dt.strftime("%Y-%m-%dT%H-00-00")
    return os.path.join([
        dir,
        time_string,
        f"gmf-{file_idx:08d}.h5"
        ])



####################################################################
# ANALYSE GMF
####################################################################


def analyze_gmf(input, cfgfile, output,
                worker=None,
                logger=None):
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

    if worker is None:
        worker = {"idx": 0, "N": 1}

    # logging
    if logger is None:
        logger = logging.getLogger(LOGGER_NAME)

    # check args
    if input is None or not os.path.isdir(input):
        logger.warning(f"input folder does not exist: {input}")
        return
    if output is None or not os.path.isdir(output):
        logger.warning(f"output folder does not exist: {output}")
        return
    if cfgfile is None or not os.path.isfile(cfgfile):
        logger.warning(f"config file does not exist: {cfgfile}")
        return


    # read drf data
    drf_data = drf.DigitalRFReader(input)
    # read config
    gmf_args = get_gmf_args(cfgfile)  

    # access args
    # n_range_gates = gmf_args["n_range_gates"]
    t0 = gmf_args["t0"]
    sample_rate = gmf_args["sample_rate"]
    ipp = gmf_args["ipp"]
    n_ipp = gmf_args["n_ipp"]
    num_cohints_per_file = gmf_args["num_cohints_per_file"]

    # gmf_max=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    # gmf_dc=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    # gmf_v=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    # gmf_a=np.zeros([num_cohints_per_file, n_range_gates], dtype=np.float32)
    # gmf_txp=np.zeros(num_cohints_per_file, dtype=np.float32)


    # rx, tx
    tx_bounds = list(drf_data.get_bounds(gmf_args["tx_channel"]))
    rx_bounds = list(drf_data.get_bounds(gmf_args["rx_channel"]))

    def to_str(value):
        return unix2datestr(value/sample_rate)

    tx_str = str([to_str(e) for e in tx_bounds])
    rx_str = str([to_str(e) for e in rx_bounds])

    logger.info(f"TX bounds: {tx_str}")
    logger.info(f"RX bounds: {rx_str}")

    # TODO - sanity check bounds

    # bounds
    bounds = list(drf_data.get_bounds(gmf_args["rx_channel"]))
    if t0 != None:
        bounds[0] = int(t0*sample_rate)
    logger.info(f"bounds {bounds}")

    # process tasks
    n_tasks = int(np.floor((bounds[1]-bounds[0])/(ipp*n_ipp))/num_cohints_per_file)
    logger.info(f"n_tasks {n_tasks}")

    # TODO - sanity check n_tasks

    #for task_idx in get_tasks(worker, n_tasks):
        #logger.info(f"process {task_idx}")
        ##file_idx = task_idx*ipp*n_ipp*num_cohints_per_file + bounds[0]        
        #outfile = get_filepath(output, file_idx, sample_rate)
        
        #if not os.path.isfile(outfile):
        #    logger.info(f"write {outfile}")
        
            #for i in range(num_cohints_per_file):
            #    i0 = file_idx + i*ipp*n_ipp
                # process gmf
                #gmf_max[i,:], gmf_dc[i,:], gmf_v[i,:], gmf_a[i,:], gmf_txp[i] = g.analyze_ipps(drf_data,i0,conf)
                #rgi=np.argmax(gmf_max[i,:])

            # """ 
            # """ ho=h5py.File(fname,"w")
            # ho["gmf"]=gmf_max
            # ho["gmf_dc"]=gmf_dc
            # ho["a"]=gmf_a
            # ho["v"]=gmf_v
            # ho["tx_pwr"]=gmf_txp
            # ho["i0"]=i0
            # ho.close() """
 #"""

    

####################################################################
# SCRIPT ENTRY POINT
####################################################################

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script analyzing eiscat drf data.",
        usage= """
        %(prog)s [options] input config -o output_folder
        
        EXAMPLE:

        %(prog)s leo_bpark_2.1u_NO@uhf/drf/ cfg/myconfig.ini
        
        """
    )

    # Add the arguments
    parser.add_argument("input", help="Path to source directory")
    parser.add_argument("cfgfile", help="Path to config file")
    parser.add_argument("output", help="Path to output directory")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # logging
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, args.log_level))

    # worker

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        class COMM_WORLD:
            rank = 0
            size = 1
        comm = COMM_WORLD()

    worker = {
        "idx": comm.rank,
        "N": comm.size
    }

    analyze_gmf(
        args.input, 
        args.cfgfile,
        args.output,
        worker = worker,
        logger = logger
    )



####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()
