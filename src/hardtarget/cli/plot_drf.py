#!/usr/bin/env python3
import matplotlib.pyplot as plt

from hardtarget import plotting
from hardtarget.drf_utils import load_hardtarget_drf
from .commands import add_command


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with Digital_RF data")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--axis_units", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--keep-tx", action="store_true")
    parser.add_argument(
        "--clutter_removal",
        type=float,
        default=0,
        help="Clutter removal at start of RX signal or after end of TX signal,\
             whichever comes last, in seconds",
    )
    return parser


def main(args):
    reader, params = load_hardtarget_drf(args.path)

    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)
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
        clutter_removal=args.clutter_removal,
        keep_tx=args.keep_tx,
    )

    plt.show()


add_command(
    name="plot_drf",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script for plotting drf raw data.",
    ),
)
