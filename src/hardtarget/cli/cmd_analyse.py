"""
The CLI Analyse functionality, abstracts the analyse functionality to a user friendly CLI interface
"""

import argparse
import logging

import hardtarget.utils.global_mpi
from hardtarget.analyse import analyse
from hardtarget.matched_filter import get_available_libs, get_default_method
from hardtarget.types.types import Job
from hardtarget.utils.profiling import get_logging_level

from .commands import add_command

DEFAULT_IMPL, DEFAULT_METHOD = get_default_method()


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds mandatory and optional positional arguments to the parser."""

    parser.add_argument("rx", help="path to source directory with rx data")
    parser.add_argument("--rxchnl", help="RX channel")
    parser.add_argument("--tx", help="path to source directory with tx data")
    parser.add_argument("--txchnl", help="TX channel")
    parser.add_argument("--config", help="path to config file for GMF processing")
    parser.add_argument("-o", "--output", default=".", help="path to output directory")
    parser.add_argument("-p", "--progress", action="store_true", help="enable progress bar")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--clobber", action="store_true", help="override outputs")
    parser.add_argument(
        "-m",
        "--method",
        help="GMF method",
        choices=[
            "fdpt",
            "fgmf",
            "grid-fast-gmf ",
            "grid-fast-no-reduce",
            "fgmf",
            "grid-fast-dpt",
            "optimize-scipy-gmf",
            "optimize-grid-gmf",
        ],
        default=DEFAULT_METHOD,
    )
    parser.add_argument(
        "-i",
        "--implementation",
        choices=["numpy", "c", "cuda"],
        help="GMF implementation",
        default=DEFAULT_IMPL,
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Analyse CLI"""

    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(get_logging_level(args.verbose))

    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)

    # import mpi (in case script is run by mpi)
    comm = hardtarget.utils.global_mpi.import_mpi()

    # job
    job = Job(idx=comm.rank, N=comm.size)

    # process
    results = analyse(
        path=args.rx,
        rx_channel=args.rxchnl,
        config=args.config,
        job=job,
        method=args.method,
        implementation=args.implementation,
        clobber=args.clobber,
        output=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
        relative_time=args.relative_time,
        progress=args.progress,
        logger=logger,
    )

    logger.info(f"produced {len(results['files'])} files")


add_command(
    name="analyse",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script analyzing eiscat drf data.",
        usage=f"""

    %(prog)s rx --rxchnl --config config_file -o output_folder

EXAMPLE:

    %(prog)s  leo_bpark_2.1u_NO@uhf/drf/ uhf

Available method implementations:
{get_available_libs(indent=" " * 4)}

        """,
    ),
)
