import logging

from . import eiscat2drf
from .commands import add_command


def parser_build(parser):
    # Add the arguments
    parser.add_argument(
        "input",
        help="Path to source directory, assumes folder structure 'input/2*/*.mat or *.mat.bz2'",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output directory, default output folder 'input/drf/uhf/'",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)",
    )
    return parser


def main(args, cli_logger):
    # Logging
    logger = logging.getLogger(eiscat2drf.LOGGER_NAME)
    logger.setLevel(getattr(logging, args.log_level))

    eiscat2drf.eiscat2drf(args.input, dstdir=args.output, logger=logger)


add_command(
    name="eiscat2drf",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script converting eiscat data to drf format",
        usage="%(prog)s [options] input -o output_folder",
    ),
)
