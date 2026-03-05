"""
Main entry for the CLI functionality
"""

import argparse
import logging
from typing import Callable

from hardtarget import __version__
from hardtarget.utils import profiling

logger = logging.getLogger(__name__)

COMMANDS: dict[str, dict] = dict()


def build_parser() -> argparse.ArgumentParser:
    """
    Build parser object from commands.
    """
    parser = argparse.ArgumentParser(description="Radar hard target processing toolbox")

    # Top level functionality
    parser.add_argument("--version", action="store_true", help="package version")

    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="count", default=0)

    # Sub level parsers
    subparsers = parser.add_subparsers(help="available command line interfaces", dest="command")

    for name, dat in COMMANDS.items():
        parser_builder, add_parser_args = dat["parser"]
        cmd_parser = subparsers.add_parser(name, **add_parser_args)
        parser_builder(cmd_parser)

    return parser


def add_command(
    name: str,
    function: Callable,
    parser_build: Callable[[argparse.ArgumentParser], argparse.ArgumentParser],
    add_parser_args: dict[str, str] = {},
) -> None:
    """
    Add a new command.
    Used by CLI scripts in order register new commands

    Args:
        name: Name of command
        function: Function to execute
        parser_build: Available parser that should be added as a subparser
        add_parser_args: parser arguments

    """
    COMMANDS[name] = dict()
    COMMANDS[name]["function"] = function
    COMMANDS[name]["parser"] = (parser_build, add_parser_args)


def main() -> None:
    """
    Main parser.
    """
    parser = build_parser()
    args = parser.parse_args()

    if profiling:
        profiling.setup_loggers(stdout=True, verbosity=args.verbose)

    if args.command is None:
        # Handle non-commands
        if args.version:
            print(__version__)
            exit()
    else:
        cmd_function = COMMANDS[args.command]["function"]
        logger.info(f"Executing command {args.command}")
        cmd_function(args)
