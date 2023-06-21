#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from hardtarget.analysis import analyze_gmf
from hardtarget.config import load_gmf_params

LOGGER_NAME = "analyse_gmf"

####################################################################
# SCRIPT ENTRY POINT
####################################################################


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script analyzing eiscat drf data.",
        usage="""
        %(prog)s [options] input config -o output_folder

        EXAMPLE:

        %(prog)s leo_bpark_2.1u_NO@uhf/drf/ cfg/myconfig.ini

        """,
    )

    # Add the arguments
    parser.add_argument("input", help="Path to source directory")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-o", "--output", help="Path to output directory")
    parser.add_argument("--clobber", action="store_true", help="Override outputs")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)",
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

    job = {"idx": comm.rank, "N": comm.size}

    # output
    output = args.output
    if output is None:
        output = "."
    output = Path(output)

    # read config file
    if not Path(args.config).is_file():
        logger.warning(f"config file does not exist: {args.config}")
        return
    gmf_params = load_gmf_params(args.config)

    # create task
    task = {
        "job": job,
        "logger": logger,
        "input": args.input,
        "output": output,
        "gmf_params": gmf_params,
        "clobber": args.clobber
    }

    print(args.clobber)

    # process
    ok, results = analyze_gmf.process(task)
    logger.info(f"produced {len(results['files'])} files")


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()
