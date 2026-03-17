"""
The CLI Plot Raw functionality, abstracts the raw data plot functionality to a user friendly CLI interface.
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from radardef import RadarDef
from radardef.types import SourceFormat

from hardtarget import plotting


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds mandatory and optional positional arguments to the parser."""

    parser.add_argument("path", help="path to source directory with raw data")
    parser.add_argument("-s", "--start_time", default=None, type=int)
    parser.add_argument("-e", "--end_time", default=None, type=int)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--axis_units", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--keep-tx", action="store_true")
    parser.add_argument("--monostatic", action="store_true")
    parser.add_argument(
        "--start-range",
        default=None,
        help="Desired starting range in given unit, can have negative values",
    )
    parser.add_argument(
        "--end-range",
        default=None,
        help="Desired ending range in given unit",
    )
    parser.add_argument(
        "-u",
        "--unit",
        choices=["sample", "m", "km", "R_E", "LD", "AU"],
        help="Unit for start and end ranges, default [km]",
        default="km",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    "Plot Raw Data CLI"

    with tempfile.TemporaryDirectory() as converted_data_path:
        radar_def = RadarDef()
        source_format = radar_def.get_source_format(args.path)
        if source_format is not SourceFormat.UNKNOWN:
            # Source format is not unknown, thus it is unconverted file
            target_formats = radar_def.available_target_formats(source_format)
            converted_files = radar_def.convert(args.path, target_formats[0], converted_data_path)
            if converted_files is None:
                raise Exception(f"Not possible to convert the file:  {args.path}")
            filepath = converted_files[0]
        else:
            # File is already converted
            filepath = Path(args.path)

        data_loader = radar_def.load_data(filepath)
        if data_loader is None:
            raise Exception(f"Not possible to load the given file: {args.path}")

        if args.start_range is not None:
            args.start_range = float(args.start_range)
        if args.end_range is not None:
            args.end_range = float(args.end_range)

        if data_loader is not None:
            fig, ax = plt.subplots()
            ax, handles = plotting.rti(
                ax,
                data_loader=data_loader,
                start_time=args.start_time,
                end_time=args.end_time,
                relative_time=args.relative_time,
                axis_units=args.axis_units,
                log=args.log,
                start_range_gate=args.start_range,
                end_range_gate=args.end_range,
                range_gate_unit=args.unit.strip().lower(),
                monostatic=args.monostatic,
                keep_tx=args.keep_tx,
            )

            plt.show()
        else:
            raise Exception(f"Not possible to load the given file: {args.path}")
