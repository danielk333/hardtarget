import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import gridspec

from hardtarget import plotting
from hardtarget.constants import AnalysisMethod
from hardtarget.plotting.load_data import load_analysed_data


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds mandatory and optional positional arguments to the parser."""

    parser.add_argument("path", help="path to analysed data")
    parser.add_argument("-s", "--start_time", default=None, type=str)
    parser.add_argument("-e", "--end_time", default=None, type=str)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--detection_limit", type=float, default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    plot_analysed_data(
        args.path,
        args.start_time,
        args.end_time,
        args.relative_time,
        args.chunk_size,
        args.detection_limit,
    )


def plot_analysed_data(
    path: Path,
    start_time: Optional[int | float | str] = None,
    end_time: Optional[int | float | str] = None,
    relative_time: bool = False,
    chunk_size: Optional[int] = None,
    detection_limit: Optional[float] = None,
) -> None:

    data_generator = load_analysed_data(  # type: ignore[var-annotated]
        data_dir=path,
        start_time=float(start_time) if relative_time else start_time,
        end_time=float(end_time) if relative_time else end_time,
        relative_time=relative_time,
        chunk_size=chunk_size,
    )

    for data in data_generator:
        out, exp, cfg, pro = data

        match pro.method:
            case AnalysisMethod.echo_search:
                fig, ax = plt.subplots(2, 2)
                plotting.plot_echo_search(ax, exp, out, limit=detection_limit)
                plt.show()
            case AnalysisMethod.direction_of_arrival:
                fig, ax = plt.subplots(3, 2)
                ax = plotting.plot_direction_of_arrival(ax, out, pro, detection_limit)
                plt.show()
            case AnalysisMethod.target_estimation:
                fig, axes = plt.subplots(2, 2)
                plotting.target_estimation_plots.plot_peaks(
                    axes,
                    out,
                    exp,
                    cfg,
                    pro,
                    snr_dB_limit=detection_limit,
                )
                if detection_limit:
                    fig, axes = plt.subplots(2, 3)
                    plotting.target_estimation_plots.plot_detections(
                        axes,
                        out,
                        exp,
                        cfg,
                        pro,
                        snr_dB_limit=detection_limit,
                    )
                fig = plt.figure()
                gs = gridspec.GridSpec(2, 2, figure=fig)
                axes = [
                    fig.add_subplot(gs[0, :]),
                    fig.add_subplot(gs[1, 0]),
                    fig.add_subplot(gs[1, 1]),
                ]
                plotting.target_estimation_plots.plot_map(
                    axes,
                    out,
                    exp,
                    cfg,
                    pro,
                )
                plt.show()
            case AnalysisMethod.optimize:
                fig, axes = plt.subplots(2, 2)
                plotting.optimization_plots.plot_optimization_peaks(
                    axes,
                    out,
                    exp,
                    cfg,
                    pro,
                    snr_dB_limit=detection_limit,
                )

                plt.show()
