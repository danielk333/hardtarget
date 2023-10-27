import logging

from . import convert_eiscat
from .commands import add_command

logger = logging.getLogger(__name__)

SOURCES = {
    "eiscat": {
        "main": convert_eiscat.main,
        "parser_build": convert_eiscat.parser_build,
        "add_parser_args": {
            "description": "Script converting eiscat data to drf format",
            "usage": "%(prog)s [options] input -o output_folder",
        },
    },
}


def parser_build(parser):
    subparsers = parser.add_subparsers(help="Available source formats", dest="source")
    subparsers.required = True

    for source in SOURCES:
        cmd_parser = subparsers.add_parser(source, **SOURCES[source]["add_parser_args"])
        parser_builder = SOURCES[source]["parser_build"]
        parser_builder(cmd_parser)

    return parser


def main(args, cli_logger):
    function = SOURCES[args.source]["main"]
    logger.info(f"Executing command {args.command}")

    function(args, logger)


add_command(
    name="convert",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Tool to convert different radar output formats to a unified digital RF definition",
    ),
)
