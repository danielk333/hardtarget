import logging
from hardtarget.radars.eiscat import download
from hardtarget.profiling import get_logging_level


def parser_build(parser):
    # Add individual arguments
    parser.add_argument('day', help='Experiment day, e.g. "20210412"')
    parser.add_argument('mode', help='Experiment mode, e.g. "leo_bpark_2.1u_NO"')
    parser.add_argument('instrument', help='Experiment instrument, e.g. "uhf|32m|42m"')
    parser.add_argument(
        'dst',
        default=".",
        help='destination folder'
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="show progressbar for download",
    )
    parser.add_argument(
        "--wget",
        action="store_true",
        help="use wget instead of requests for download",
    )

    return parser


def main(args):
    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(get_logging_level(args.verbose))
    # download
    download(args.day, args.mode, args.instrument, args.dst,
             logger=logger, progress=args.progress, wget=args.wget)
