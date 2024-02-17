import logging
from .commands import add_command
from . import check_cuda

logger = logging.getLogger(__name__)

SOURCES = {
    "cuda": {
        "main": check_cuda.main,
        "parser_build": check_cuda.parser_build,
        "add_parser_args": {
            "description": "Test target",
            "usage": "%(prog)s [options] path",
        },
    }
}


def parser_build(parser):
    global SOURCES
    subparsers = parser.add_subparsers(help="hardtarget check types", dest="checktype")
    subparsers.required = True
    for source in SOURCES:
        cmd_parser = subparsers.add_parser(source, **SOURCES[source]["add_parser_args"])
        parser_builder = SOURCES[source]["parser_build"]
        parser_builder(cmd_parser)
    return parser


def main(args):
    global SOURCES
    function = SOURCES[args.checktype]["main"]
    logger.info(f"Executing command {args.command} {args.checktype}")
    function(args)


add_command(
    name="check",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Hardtarget check tools",
    ),
)

