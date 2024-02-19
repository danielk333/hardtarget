import logging
from .commands import add_command
from . import inspect_gmf
from . import inspect_drf

logger = logging.getLogger(__name__)

SOURCES = {
    "gmf": {
        "main": inspect_gmf.main,
        "parser_build": inspect_gmf.parser_build,
        "add_parser_args": {
            "description": "inspect GMF file",
            "usage": "%(prog)s [options] path",
        },
    },
    "drf": {
        "main": inspect_drf.main,
        "parser_build": inspect_drf.parser_build,
        "add_parser_args": {
            "description": "inspect DRF file",
            "usage": "%(prog)s [options] path",
        },
    }
}


def parser_build(parser):
    global SOURCES
    subparsers = parser.add_subparsers(help="hardtarget file types", dest="filetype")
    subparsers.required = True
    for source in SOURCES:
        cmd_parser = subparsers.add_parser(source, **SOURCES[source]["add_parser_args"])
        parser_builder = SOURCES[source]["parser_build"]
        parser_builder(cmd_parser)
    return parser


def main(args):
    global SOURCES
    function = SOURCES[args.filetype]["main"]
    logger.info(f"Executing command {args.command} {args.filetype}")
    function(args)


add_command(
    name="inspect",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Hardtarget file inspection tools",
    ),
)
