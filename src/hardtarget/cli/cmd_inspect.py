"""
The CLI Inspect functionality, gathers the inspect functionality for both the analysed and raw format.
"""

import argparse
import logging

from hardtarget.types import ParserArgs, SubParser

from . import inspect_mf, inspect_raw_data
from .commands import add_command

logger = logging.getLogger(__name__)

SOURCES = {
    "mf": SubParser(
        main=inspect_mf.main,
        parser_build=inspect_raw_data.parser_build,
        parser_args=ParserArgs(description="inspect MF file", usage="%(prog)s [options] path"),
    ),
    "raw": SubParser(
        main=inspect_raw_data.main,
        parser_build=inspect_raw_data.parser_build,
        parser_args=ParserArgs(
            description="inspect raw file",
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
    """Inspect CLI"""
    function = SOURCES[args.filetype].main
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
