#!/usr/bin/env python

import logging
from hardtarget.analysis import compute_gmf

from .commands import add_command


LOGGER_NAME = "hardtarget.analysis.analyze_gmf"


####################################################################
# SCRIPT ENTRY POINT
####################################################################


def parser_build(parser):
    # Add the arguments
    parser.add_argument("rx", help="Path to source directory with rx data")
    parser.add_argument("rxchnl", help="RX channel")
    parser.add_argument("--tx", help="Path to source directory with tx data")
    parser.add_argument("--txchnl", help="TX channel")
    parser.add_argument("--config", help="Path to config file for GMF processing")
    parser.add_argument("-o", "--output", default=".", help="Path to output directory")
    parser.add_argument("--progress", action="store_true", help="Progress bar")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--clobber", action="store_true", help="Override outputs")
    parser.add_argument(
        "-g",
        "--gmflib",
        choices=["numpy", "c", "cuda"],
        help="GMF implementation",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)",
    )
    return parser


def main(args):
    # logging
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, args.log_level))

    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)

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

    # rx and tx
    rx = (args.rx, args.rxchnl)
    if args.tx is None:
        args.tx = args.rx
    if args.txchnl is None:
        args.txchnl = args.rxchnl
    tx = (args.tx, args.txchnl)

    # process
    results = compute_gmf(
        rx,
        tx,
        config=args.config,
        job=job,
        gmflib=args.gmflib,
        clobber=args.clobber,
        output=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
        relative_time=args.relative_time,
        progress=args.progress,
        progress_position=comm.rank,
    )

    logger.info(f"produced {len(results['files'])} files")


add_command(
    name="gmf",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script analyzing eiscat drf data.",
        usage="""
        %(prog)s rx rxchnl --config config_file -o output_folder

        EXAMPLE:

        %(prog)s  leo_bpark_2.1u_NO@uhf/drf/ uhf

        """,
    ),
)

####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()
