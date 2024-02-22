import logging
import numpy as np
import scipy.constants as constants
from .commands import add_command
from .utils import SI_to_unit, unit_to_SI
from hardtarget.drf_utils import load_hardtarget_drf


logger = logging.getLogger(__name__)


#################################################
# CHECK RANGE-GATES
#################################################

def range_gates_parser_build(parser):
    parser.add_argument("path", help="path to source directory with DRF data")
    parser.add_argument("--start-range", "-s", default=None, help="Desired starting range in given unit")
    parser.add_argument("--end-range", "-e", default=None, help="Desired ending range in given unit")
    parser.add_argument(
        "-u", "--unit",
        choices=["m", "km", "R_E", "LD", "AU"],
        help="Unit for start and end ranges",
        default="km",
    )
    return parser


def range_gates_main(args):
    _, meta = load_hardtarget_drf(args.path)

    sample_rate = meta["sample_rate"]
    rx_start = meta["rx_start"]
    rx_end = meta["rx_end"]
    tx_start = meta["tx_start"]

    T_rx_start_samp = np.round(rx_start * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(rx_end * 1e-6 * sample_rate).astype(np.int64)
    T_tx_start_samp = np.round(tx_start * 1e-6 * sample_rate).astype(np.int64)

    il0_rgs_min = T_tx_start_samp + 1
    il0_rgs_max = T_rx_end_samp
    rgs_min = il0_rgs_min - (T_tx_start_samp + 1)
    rgs_max = il0_rgs_max - (T_tx_start_samp + 1)
    rgs_min_sec = (il0_rgs_min - T_tx_start_samp)/sample_rate
    rgs_min_km = rgs_min_sec*constants.c*1e-3
    rgs_max_sec = (il0_rgs_max - T_tx_start_samp)/sample_rate
    rgs_max_km = rgs_max_sec*constants.c*1e-3

    print(f"DRF '{args.path}':")
    print(f" - Minimum range gate IL0 sample (range-gate {rgs_min}): {il0_rgs_min} ({rgs_min_km} km)")
    print(f" - Maximum range gate IL0 sample (range-gate {rgs_max}): {il0_rgs_max} ({rgs_max_km} km)")
    if args.start_range is not None:
        args.start_range = unit_to_SI(float(args.start_range), args.unit.lower())
        il0_rg0 = sample_rate*args.start_range/constants.c + T_tx_start_samp
        il0_rg0 = np.round(il0_rg0).astype(np.int64)
        rg0_sec = (il0_rg0 - T_tx_start_samp)/sample_rate
        rg0_unit = SI_to_unit(rg0_sec*constants.c, args.unit.lower())

        rg0 = il0_rg0 - (T_tx_start_samp + 1)
        print(f" - Requested start range ({rg0_unit} {args.unit}): IL0 sample {il0_rg0} (range-gate {rg0})")
        assert il0_rg0 <= T_rx_end_samp, "start range gate cannot be after than RX end"
        assert il0_rg0 > T_rx_start_samp, "start range gate cannot be before than RX start"

    if args.end_range is not None:
        args.end_range = unit_to_SI(float(args.end_range), args.unit.lower())
        il0_rg1 = sample_rate*args.end_range/constants.c + T_tx_start_samp
        il0_rg1 = np.round(il0_rg1).astype(np.int64)
        rg1_sec = (il0_rg1 - T_tx_start_samp)/sample_rate
        rg1_unit = SI_to_unit(rg1_sec*constants.c, args.unit.lower())

        rg1 = il0_rg1 - (T_tx_start_samp + 1)
        print(f" - Requested end range ({rg1_unit} {args.unit}): IL0 sample {il0_rg1} (range-gate {rg1})")
        assert il0_rg1 <= T_rx_end_samp, "end range gate cannot be after than RX end"
        assert il0_rg1 > T_rx_start_samp, "end range gate cannot be before than RX start"


#################################################
# CUDA CHECK COMMAND
#################################################

def cuda_parser_build(parser):
    return parser


def cuda_main(args):
    try:
        import hardtarget.gmf.gmf_cuda as gcu
        gcu.print_cuda_devices()
    except ImportError as e:
        print(e)


#################################################
# COMMANDS
#################################################

SOURCES = {
    "cuda": {
        "main": cuda_main,
        "parser_build": cuda_parser_build,
        "add_parser_args": {
            "description": "Test target",
            "usage": "%(prog)s [options] path",
        },
    },
    "range-gates": {
        "main": range_gates_main,
        "parser_build": range_gates_parser_build,
        "add_parser_args": {
            "description": "Check the available range gates (two-way range) of the target DRF",
            "usage": "%(prog)s [options] path",
        },
    }
}


def parser_build(parser):
    global SOURCES
    subparsers = parser.add_subparsers(help="hardtarget check types", dest="checktype")
    subparsers.required = True
    for source in SOURCES:
        cmd_parser = subparsers.add_parser(source, **SOURCES[source]["add_parser_args"])
        parser_builder = SOURCES[source]["parser_build"]
        parser_builder(cmd_parser)
    return parser


def main(args):
    global SOURCES
    function = SOURCES[args.checktype]["main"]
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