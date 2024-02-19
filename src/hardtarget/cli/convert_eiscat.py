import logging
from hardtarget.convert.eiscat import eiscat_convert


####################################################################
# SCRIPT ENTRY POINT
####################################################################

def parser_build(parser):
    # Add the arguments
    parser.add_argument(
        "input",
        help="path to source directory, assumes folder structure 'input/2*/*.mat or *.mat.bz2'",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to output directory, default output folder 'input/drf/uhf/'",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set the log level (default: INFO)",
    )
    parser.add_argument(
        "-c",
        "--compression",
        choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        default="0",
        help="set the compression level (0-9) (default: 0)",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="show progressbar for conversion",
    )
    return parser


def main(args):
    compression_level = int(args.compression)
    # Logging
    logger = logging.getLogger("eiscat_convert")
    logger.setLevel(getattr(logging, args.log_level))
    eiscat_convert(
        args.input,
        logger,
        dstdir=args.output,
        compression_level=compression_level,
        progress=args.progress,
    )
