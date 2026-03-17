"""
The CLI Inspect Raw functionality, abstracts the inspect functionality for raw data files to a user friendly
CLI interface.
"""

import argparse
import pprint
import tempfile
from collections import OrderedDict
from pathlib import Path

from radardef import RadarDef
from radardef.types import SourceFormat

from hardtarget.utils.time_conversion import str_from_ts


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Define argparse sub parser."""
    parser.add_argument("path", help="path to source directory with raw data")
    return parser


def main(args: argparse.Namespace) -> None:
    """Inspect Raw data CLI"""

    with tempfile.TemporaryDirectory() as converted_data_path:
        radar_def = RadarDef()
        source_format = radar_def.get_source_format(args.path)
        if source_format is not SourceFormat.UNKNOWN:
            target_formats = radar_def.available_target_formats(source_format)
            converted_files = radar_def.convert(args.path, target_formats[0], converted_data_path)
            if converted_files is None:
                raise Exception(f"Not possible to convert the file:  {args.path}")
            filepath = converted_files[0]
        else:
            filepath = Path(args.path)

        data_loader = radar_def.load_data(filepath)
        if data_loader is None:
            raise Exception(f"Not possible to load the given file: {args.path}")

        d = []
        channels = data_loader.channels
        for chnl in channels:
            samp_start, samp_end = data_loader.bounds(chnl)
            sample_rate = data_loader.meta.experiment.sample_rate
            dt_start = str_from_ts(data_loader.meta.bounds.ts_start_usec * 1e-6)
            dt_end = str_from_ts(data_loader.meta.bounds.ts_end_usec * 1e-6)

            d.append(
                OrderedDict(
                    channel=str(chnl),
                    sample_rate=sample_rate,
                    start=f"{dt_start}",
                    end=f"{dt_end}",
                    sample_bounds=[int(samp_start), int(samp_end)],
                    samples=int(samp_end - samp_start),
                )
            )

        pprint.pprint({"Channels": d, "Experiment": data_loader.meta.experiment._asdict()})
