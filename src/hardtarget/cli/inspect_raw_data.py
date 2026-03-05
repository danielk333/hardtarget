"""
The CLI Inspect Raw functionality, abstracts the inspect functionality for raw data files to a user friendly
CLI interface.
"""

import argparse
import datetime
import pprint
import tempfile
from collections import OrderedDict
from pathlib import Path

from radardef import RadarDef
from radardef.types import SourceFormat

from hardtarget.types.types import Bounds


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
            t_start, t_end = data_loader.bounds(chnl)
            bounds = Bounds(t_start, t_end)
            sample_rate = 1 / (data_loader.meta.experiment.t_ipp_usec * 1e-6)
            dt0 = datetime.datetime.fromtimestamp(bounds.start / sample_rate, datetime.timezone.utc)
            dt1 = datetime.datetime.fromtimestamp(bounds.end / sample_rate, datetime.timezone.utc)
            mega_samples = (bounds.end - bounds.start) * 1e-6

            d.append(
                OrderedDict(
                    channel=chnl,
                    start=f"{dt0}",  # dt.strftime("%Y-%m-%dT%H:%M:%S")
                    end=f"{dt1}",
                    bounds=bounds,
                    mega_samples=mega_samples,
                )
            )

        pprint.pprint({"Data": d, "Exp": data_loader.meta.experiment._asdict})
