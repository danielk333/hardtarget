#!/usr/bin/env python

import os
import argparse
import logging
import configparser
from hardtarget.analysis import analyze_gmf

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

    # job
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        class COMM_WORLD:
            rank = 0
            size = 1
        comm = COMM_WORLD()

    job = {
        "idx": comm.rank,
        "N": comm.size
    }

    # read config file
    if args.cfgfile is None or not os.path.isfile(args.cfgfile):
        logger.warning(f"config file does not exist: {args.cfgfile}")
        return    
    gmf_args = get_gmf_args(args.cfgfile)  

    # create task
    task = {
        "job": job,
        "logger": logger,
        "ipp": gmf_args["ipp"],
        "n_ipp": gmf_args["n_ipp"],
        "num_cohints_per_file": gmf_args["num_cohints_per_file"],
        "t0":  gmf_args.get("t0", None),
        "n_range_gates": gmf_args["n_range_gates"],
        "input": args.input,
        "output": args.output 
    }

    # process
    result = analyze_gmf.process(task)




####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()
