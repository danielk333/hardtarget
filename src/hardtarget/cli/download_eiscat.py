import logging
from hardtarget.experiments import eiscat_download
from hardtarget.profiling import get_logging_level


def parser_build(parser):
    # Add individual arguments
    parser.add_argument('date', help='Eiscat experiment date, e.g. "20210412"')
    parser.add_argument('name', help='Eiscat experiment name, e.g. "leo_bpark_2.1u_NO"')
    parser.add_argument('type', help='Eiscat experiment type, e.g. "uhf|32|42"')
    parser.add_argument('--update', action='store_true', help='If true, contents of destination will be updated with downloaded data')
    parser.add_argument(
        '-o', 
        '--output', 
        default=".", 
        help='destination folder'
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="show progressbar for download",
    )
    return parser


def main(args):
    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(get_logging_level(args.verbose))
    # download
    eiscat_download.download(args.date, args.name, args.type, args.output, 
                             logger=logger, update=args.update, progress=args.progress)
