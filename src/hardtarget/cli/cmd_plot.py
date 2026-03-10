"""
The CLI Plot functionality,  gathers the plot functionality for both the analysed and raw format.
"""

import argparse
import logging

from hardtarget.types import ParserArgs, SubParser

from . import plot_analysed_data, plot_raw_data
from .commands import add_command

logger = logging.getLogger(__name__)

SOURCES = {
    "analysis": SubParser(
        main=plot_analysed_data.main,
        parser_build=plot_analysed_data.parser_build,
        parser_args=ParserArgs(
            description="Plot analysed data ",
            usage="%(prog)s [options] path",
        ),
    ),
    "raw": SubParser(
        main=plot_raw_data.main,
        parser_build=plot_raw_data.parser_build,
        parser_args=ParserArgs(
            description="Plot raw data",
            usage="%(prog)s [options] path",
        ),
    ),
}


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds mandatory and optional positional arguments to the parser."""
    subparsers = parser.add_subparsers(help="hardtarget file types", dest="filetype")
    subparsers.required = True
    for source in SOURCES:
        cmd_parser = subparsers.add_parser(source, **SOURCES[source].parser_args)
        parser_builder = SOURCES[source].parser_build
        parser_builder(cmd_parser)
    return parser


def main(args: argparse.Namespace) -> None:
    """Plot CLI"""
    function = SOURCES[args.filetype].main
    logger.info(f"Executing command {args.command} {args.filetype}")
    function(args)


add_command(
    name="plot",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Hardtarget file plotting tools",
    ),
)
