#!/usr/bin/env python3
import matplotlib.pyplot as plt

from hardtarget import plotting
from .commands import add_command


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with GMF output data")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("--relative_time", action="store_true")
    return parser


def main(args, cli_logger):
    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)
    fig, ax = plt.subplots()
    ax, handles = plotting.gmf_peak(
        ax,
        args.path,
        start_time=args.start_time,
        end_time=args.end_time,
        relative_time=args.relative_time,
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
