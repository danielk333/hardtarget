#!/usr/bin/env python

import os
import argparse
import logging
import configparser
from hardtarget.analysis import analyze_gmf
from hardtarget.analysis import analyze_params

LOGGER_NAME = "analyse_gmf"

####################################################################
# LOAD GMF PARAMS FROM CONFIG
####################################################################

def load_gmf_params(config_file):
    """
    Load a gmf config file into to a dictionary
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    d = {}
    if 'config' in config:
        for key, value in config['config'].items():
            # Convert values to specific types
            if key in analyze_params.INT_PARAM_KEYS:
                d[key] = int(value)
            elif key in analyze_params.BOOL_PARAM_KEYS:
                d[key] = config.getboolean('config', key)
            elif key in analyze_params.FLOAT_PARAM_KEYS:
                d[key] = float(value)
            else:                
                # string
                d[key] = value.strip('"')
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
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-o", "--output", help="Path to output directory")
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

    # output
    output = args.output
    if output is None:
        output = "."
    output = os.path.abspath(output)



    # read config file
    if args.config is None:
        # guess uhf
        gmf_params = {"rx_channel": "uhf", "tx_channel": "uhf"}
    else:
        if not os.path.isfile(args.config):
            logger.warning(f"config file does not exist: {args.config}")
            return    
        gmf_params = load_gmf_params(args.config)  

    # create task
    task = {
        "job": job,
        "logger": logger,
        "input": args.input,
        "output": output,
        "gmf_params": gmf_params 
    }

    # process
    ok, results = analyze_gmf.process(task)
    logger.info(f"produced {len(results['files'])} files")


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()