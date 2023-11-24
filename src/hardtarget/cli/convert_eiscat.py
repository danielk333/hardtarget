#!/usr/bin/env python
import logging
from hardtarget.convert.eiscat import eiscat_convert


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
    parser.add_argument(
        "-c",
        "--compression",
        choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        default="0",
        help="Set the compression level (0-9) (default: 0)",
    )

    return parser


def main(args):
    compression_level = int(args.compression)
    # Logging
    logger = logging.getLogger("eiscat_convert")
    logger.setLevel(getattr(logging, args.log_level))
    eiscat_convert(args.input,
                   logger,
                   dstdir=args.output,
                   compression_level=compression_level)


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
    main(args)
