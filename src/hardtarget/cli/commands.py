import argparse
import logging

from hardtarget import profiling

logger = logging.getLogger(__name__)

COMMANDS = dict()


def build_parser():
    parser = argparse.ArgumentParser(description="Radar hard target processing toolbox")

    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="count", default=0)

    subparsers = parser.add_subparsers(help="Available command line interfaces", dest="command")
    subparsers.required = True

    for name, dat in COMMANDS.items():
        parser_builder, add_parser_args = dat["parser"]
        cmd_parser = subparsers.add_parser(name, **add_parser_args)
        parser_builder(cmd_parser)

    return parser


def add_command(name, function, parser_build, add_parser_args={}):
    global COMMANDS
    COMMANDS[name] = dict()
    COMMANDS[name]["function"] = function
    COMMANDS[name]["parser"] = (parser_build, add_parser_args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    profiling.setup_loggers(stdout=True, verbosity=args.verbose)

    function = COMMANDS[args.command]["function"]
    logger.info(f"Executing command {args.command}")

    function(args)
