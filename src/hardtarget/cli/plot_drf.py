from hardtarget import plotting
import matplotlib.pyplot as plt
from hardtarget.drf_utils import load_hardtarget_drf


def parser_build(parser):
    parser.add_argument("path", help="path to source directory with Digital_RF data")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
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
        "-u", "--unit",
        choices=["sample", "m", "km", "R_E", "LD", "AU"],
        help="Unit for start and end ranges, default [km]",
        default="km",
    )
    return parser


def main(args):
    reader, params = load_hardtarget_drf(args.path)

    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)

    if args.start_range is not None:
        args.start_range = float(args.start_range)
    if args.end_range is not None:
        args.end_range = float(args.end_range)

    fig, ax = plt.subplots()
    ax, handles = plotting.rti(
        ax,
        reader,
        params,
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
