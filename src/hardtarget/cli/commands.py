import argparse
import logging
from hardtarget import profiling, __version__

logger = logging.getLogger(__name__)

COMMANDS = dict()


def build_parser():
    """
    Build parser object from commands.
    """
    parser = argparse.ArgumentParser(description="Radar hard target processing toolbox")

    # Top level functionality
    parser.add_argument('--version', action='store_true', 
                        help='Package version')

    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="count", default=0)

    # Sub level parsers
    subparsers = parser.add_subparsers(help="Available command line interfaces", dest="command")

    for name, dat in COMMANDS.items():
        parser_builder, add_parser_args = dat["parser"]
        cmd_parser = subparsers.add_parser(name, **add_parser_args)
        parser_builder(cmd_parser)

    return parser


def add_command(name, function, parser_build, add_parser_args={}):
    """
    Add a new command.
    Used by CLI scripts in order register new commands
    """
    global COMMANDS
    COMMANDS[name] = dict()
    COMMANDS[name]["function"] = function
    COMMANDS[name]["parser"] = (parser_build, add_parser_args)


def main():
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


