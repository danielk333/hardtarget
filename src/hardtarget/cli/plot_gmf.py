import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hardtarget.plotting import gmf
from hardtarget.analysis import load_gmf_out
from .commands import add_command


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with GMF output data")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--snr_dB_limit", type=float, default=15.0)
    parser.add_argument("--not_monostatic", action="store_true")
    return parser


def main(args):
    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)
    if args.chunk_size == 0:
        args.chunk_size = None

    data_generator = load_gmf_out(
        args.path,
        start_time=args.start_time,
        end_time=args.end_time,
        relative_time=args.relative_time,
        chunk_size=args.chunk_size,
    )

    for data, meta in data_generator:
        fig, axes = plt.subplots(2, 2)
        gmf.plot_peaks(
            axes,
            data,
            meta,
            monostatic=not args.not_monostatic,
            snr_dB_limit=args.snr_dB_limit,
        )

        fig, axes = plt.subplots(2, 3)
        gmf.plot_detections(
            axes,
            data,
            meta,
            monostatic=not args.not_monostatic,
            snr_dB_limit=args.snr_dB_limit,
        )

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2, figure=fig)
        axes = [
            fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
        gmf.plot_map(axes, data, meta)

        plt.show()
