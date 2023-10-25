#!/usr/bin/env python
from hardtarget.convert.eiscat_convert import eiscat_convert


####################################################################
# SCRIPT ENTRY POINT
####################################################################

def parser_build(parser):
    # Add the arguments
    parser.add_argument(
        "input",
        help="Path to source directory, assumes folder structure 'input/2*/*.mat or *.mat.bz2'",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output directory, default output folder 'input/drf/uhf/'",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)",
    )
    return parser


def main(args, cli_logger):
    eiscat_convert(args.input, cli_logger, dstdir=args.output)


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":

    import logging
    import argparse

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script converting eiscat data to drf format",
        usage="%(prog)s [options] input -o output_folder",
    )
    parser = parser_build(parser)
    # Parse the arguments
    args = parser.parse_args()

    # Logging
    logger = logging.getLogger("eiscat_convert")
    logger.setLevel(getattr(logging, args.log_level))
    main(args, logger)
