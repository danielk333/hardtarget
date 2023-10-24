#!/usr/bin/env python

import logging
from progress.bar import Bar
from pathlib import Path

from hardtarget.analysis import analyze_gmf
from hardtarget.config import load_gmf_params

from .commands import add_command


LOGGER_NAME = "analyse_gmf"


####################################################################
# SCRIPT ENTRY POINT
####################################################################

def parser_build(parser):
    # Add the arguments
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("rx", help="Path to source directory with rx data")
    parser.add_argument("rxchnl", help="RX channel")
    parser.add_argument("--tx", help="Path to source directory with tx data")
    parser.add_argument("--txchnl", help="TX channel")
    parser.add_argument("-o", "--output", help="Path to output directory")
    parser.add_argument("--progress", action="store_true", help="Progress bar")
    parser.add_argument("--clobber", action="store_true", help="Override outputs")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)",
    )
    return parser


def main(args, cli_logger):

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

    # config file
    if not Path(args.config).is_file():
        logger.warning(f"config file does not exist: {args.config}")
        return
    gmf_params = load_gmf_params(args.config)

    # create task
    task = {
        "job": job,
        "logger": logger,
        "rx": (args.rx, args.rxchnl),
        "tx": (args.tx, args.txchnl),
        "output": output,
        "gmf_params": gmf_params,
        "clobber": args.clobber
    }

    # preprocess
    ok = analyze_gmf.preprocess(task)
    if (ok):

        # progress
        if args.progress:
            progress_bar = Bar('Processing', max=len(task["job_tasks"]))

            def progress_callback(numerator, divisor):
                progress_bar.goto(numerator)

            task["progress"] = progress_callback

        # process
        ok, results = analyze_gmf.process(task)
        logger.info(f"produced {len(results['files'])} files")

        if args.process:
            progress_bar.finish()


add_command(
    name="analyze",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script analyzing eiscat drf data.",
        usage="""
        %(prog)s [options] config rx rxchnl -o output_folder

        EXAMPLE:

        %(prog)s cfg/myconfig.ini leo_bpark_2.1u_NO@uhf/drf/ uhf

        """,
    ),
)

####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()
