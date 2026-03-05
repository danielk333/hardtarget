"""
The CLI Check functionality, analyses the raw data and extracts relevant parameters for the analysis
configuration.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import scipy.constants as constants
from radardef import RadarDef
from radardef.types import SourceFormat

from hardtarget.types.types import ParserArgs, SubParser
from hardtarget.utils.range_conversion import SI_to_unit, unit_to_SI

from .commands import add_command

logger = logging.getLogger(__name__)


def range_gates_parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds mandatory and optional positional arguments to the parser."""

    parser.add_argument("path", help="path to source directory with raw data")
    parser.add_argument("--start-range", "-s", default=None, help="Desired starting range in given unit")
    parser.add_argument("--end-range", "-e", default=None, help="Desired ending range in given unit")
    parser.add_argument(
        "--clutter-range",
        "-c",
        default=None,
        help=(
            "Desired clutter location range in given unit, "
            "calculates what range-gate is complexly free of "
            "signal return from that range."
        ),
    )
    parser.add_argument(
        "-u",
        "--unit",
        choices=["m", "km", "R_E", "LD", "AU"],
        help="Unit for start and end ranges",
        default="km",
    )
    return parser


def range_gates_main(args: argparse.Namespace) -> None:
    """Check CLI"""

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
            logger.warning(f"Not possible to read data from path: {args.path}")
            return

        meta = data_loader.meta

    sample_rate = 1 / (meta.experiment.t_samp_usec * 1e-6)
    rx_start = meta.experiment.t_rx_start_usec if meta.experiment.t_rx_start_usec is not None else 0
    rx_end = meta.experiment.t_rx_end_usec if meta.experiment.t_rx_end_usec is not None else 0
    tx_start = meta.experiment.t_tx_start_usec if meta.experiment.t_tx_start_usec is not None else 0
    tx_end = meta.experiment.t_tx_end_usec if meta.experiment.t_tx_end_usec is not None else 0

    T_rx_start_samp = np.round(rx_start * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(rx_end * 1e-6 * sample_rate).astype(np.int64)
    T_tx_start_samp = np.round(tx_start * 1e-6 * sample_rate).astype(np.int64)
    T_tx_end_samp = np.round(tx_end * 1e-6 * sample_rate).astype(np.int64)

    il0_rgs_min = T_tx_start_samp + 1
    il0_rgs_max = T_rx_end_samp
    rgs_min = il0_rgs_min - (T_tx_start_samp + 1)
    rgs_max = il0_rgs_max - (T_tx_start_samp + 1)
    rgs_min_sec = (il0_rgs_min - T_tx_start_samp) / sample_rate
    rgs_min_km = rgs_min_sec * constants.c * 1e-3
    rgs_max_sec = (il0_rgs_max - T_tx_start_samp) / sample_rate
    rgs_max_km = rgs_max_sec * constants.c * 1e-3

    tx_end_rg = T_tx_end_samp - (T_tx_start_samp + 1)
    tx_end_rg_km = (tx_end_rg + 1) / sample_rate * constants.c * 1e-3

    print(f"File: '{args.path}':")
    print(f" - Minimum range gate IL0 sample (range-gate {rgs_min}): {il0_rgs_min} ({rgs_min_km} km)")
    print(f" - Maximum range gate IL0 sample (range-gate {rgs_max}): {il0_rgs_max} ({rgs_max_km} km)")
    print(f" - Range-gate at TX pulse end {tx_end_rg} ({tx_end_rg_km} km)")

    if args.clutter_range is not None:
        args.clutter_range = unit_to_SI(float(args.clutter_range), args.unit.lower())
        il0_clutter = sample_rate * args.clutter_range / constants.c + T_tx_start_samp
        T_tx_samps = T_tx_end_samp - T_tx_start_samp
        il0_no_clutter = il0_clutter + T_tx_samps
        no_clutter_sec = (il0_no_clutter - T_tx_start_samp) / sample_rate
        no_clutter_unit = SI_to_unit(no_clutter_sec * constants.c, args.unit.lower())
        rg_no_clutter = il0_no_clutter - (T_tx_start_samp + 1)
        print(
            f" - Requested clutter range-gate ({no_clutter_unit} {args.unit}): "
            f"IL0 sample {il0_no_clutter} (range-gate {rg_no_clutter})"
        )

    if args.start_range is not None:
        args.start_range = unit_to_SI(float(args.start_range), args.unit.lower())
        il0_rg0 = sample_rate * args.start_range / constants.c + T_tx_start_samp
        il0_rg0 = np.round(il0_rg0).astype(np.int64)
        rg0_sec = (il0_rg0 - T_tx_start_samp) / sample_rate
        rg0_unit = SI_to_unit(rg0_sec * constants.c, args.unit.lower())

        rg0 = il0_rg0 - (T_tx_start_samp + 1)
        print(f" - Requested start range ({rg0_unit} {args.unit}): IL0 sample {il0_rg0} (range-gate {rg0})")
        assert il0_rg0 <= T_rx_end_samp, "start range gate cannot be after than RX end"
        assert il0_rg0 > T_rx_start_samp, "start range gate cannot be before than RX start"

    if args.end_range is not None:
        args.end_range = unit_to_SI(float(args.end_range), args.unit.lower())
        il0_rg1 = sample_rate * args.end_range / constants.c + T_tx_start_samp
        il0_rg1 = np.round(il0_rg1).astype(np.int64)
        rg1_sec = (il0_rg1 - T_tx_start_samp) / sample_rate
        rg1_unit = SI_to_unit(rg1_sec * constants.c, args.unit.lower())

        rg1 = il0_rg1 - (T_tx_start_samp + 1)
        print(f" - Requested end range ({rg1_unit} {args.unit}): IL0 sample {il0_rg1} (range-gate {rg1})")
        assert il0_rg1 <= T_rx_end_samp, "end range gate cannot be after than RX end"
        assert il0_rg1 > T_rx_start_samp, "end range gate cannot be before than RX start"


def cuda_parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Specific cuda parser build"""
    return parser


def cuda_main(args: argparse.Namespace) -> None:
    """Validation of the cuda environment if present"""
    try:
        import hardtarget.matched_filter.gmf.gmf_cuda as gcu

        gcu.print_cuda_devices()
        gcu.test_cuda()
    except ImportError as e:
        print(e)


SOURCES: dict[str, SubParser] = {
    "cuda": SubParser(
        main=cuda_main,
        parser_build=cuda_parser_build,
        parser_args=ParserArgs(description="Check cuda devices and functionality", usage=""),
    ),
    "range-gates": SubParser(
        main=range_gates_main,
        parser_build=range_gates_parser_build,
        parser_args=ParserArgs(
            description="Check the available range gates (two-way range) of the target DRF", usage=""
        ),
    ),
}


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds mandatory and optional positional arguments to the parser."""
    subparsers = parser.add_subparsers(help="hardtarget check types", dest="checktype")
    subparsers.required = True
    for source in SOURCES:
        cmd_parser = subparsers.add_parser(source, **SOURCES[source].parser_args)
        parser_builder = SOURCES[source].parser_build
        parser_builder(cmd_parser)
    return parser


def main(args: argparse.Namespace) -> None:
    function = SOURCES[args.checktype].main
    logger.info(f"Executing command {args.command} {args.checktype}")
    function(args)


add_command(
    name="check",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Hardtarget check tools",
    ),
)
