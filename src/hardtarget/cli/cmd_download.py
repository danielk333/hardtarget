import logging

from . import download_eiscat
from .commands import add_command

logger = logging.getLogger(__name__)


SOURCES = {
    "eiscat": {
        "main": download_eiscat.main,
        "parser_build": download_eiscat.parser_build,
        "add_parser_args": {
            "description": "Script downloading eiscat",
            "usage": "%(prog)s [options] input -o output_folder",
        },
    },
}


def parser_build(parser):
    subparsers = parser.add_subparsers(help="available source formats", dest="source")
    subparsers.required = True

    for source in SOURCES:
        cmd_parser = subparsers.add_parser(source, **SOURCES[source]["add_parser_args"])
        parser_builder = SOURCES[source]["parser_build"]
        parser_builder(cmd_parser)

    return parser


def main(args):
    function = SOURCES[args.source]["main"]
    logger.info(f"Executing command {args.command}")
    function(args)


add_command(
    name="download",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Tool to download radar data from Eiscat",
    ),
)
