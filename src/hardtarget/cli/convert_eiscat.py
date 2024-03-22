import logging
from hardtarget.radars.eiscat import convert
from hardtarget.profiling import get_logging_level


def parser_build(parser):
    # Add the arguments
    parser.add_argument(
        "src",
        type=str,
        help="path to Eiscat raw data source directory",
    )
    parser.add_argument(
        "dst",
        type=str,
        help="path to destination directory",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="name for Hardtarget DRF folder",
        default=None,
    )

    parser.add_argument(
        "-c",
        "--compression",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        default=0,
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
    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(get_logging_level(args.verbose))
    convert(
        args.src,
        args.dst,
        name=args.name,
        logger=logger,
        compression=args.compression,
        progress=args.progress,
    )
