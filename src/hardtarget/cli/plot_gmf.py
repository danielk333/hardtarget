#!/usr/bin/env python3
import matplotlib.pyplot as plt

from hardtarget.plotting import gmf
from .commands import add_command


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with GMF output data")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=0)
    return parser


def main(args):
    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)
    if args.chunk_size == 0:
        args.chunk_size = None

    paths = gmf.collect_paths(
        args.path,
        start_time=args.start_time,
        end_time=args.end_time,
        relative_time=args.relative_time,
    )

    data_generator = gmf.yield_chunked_data(paths, chunk_size=args.chunk_size)

    for data in data_generator:
        fig, axes = plt.subplots(2, 2)
        gmf.plot_peaks(axes, data)
        fig, axes = plt.subplots(2, 2)
        gmf.plot_map(axes, data)
        plt.show()


add_command(
    name="plot_gmf",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script for plotting gmf output.",
    ),
)
