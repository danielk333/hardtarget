"""
The CLI Plot MF functionality, abstracts the mf plot functionality to a user friendly CLI interface.
"""

import argparse
from typing import Generator

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from hardtarget.matched_filter.types import (
    ExtendedTargetEstimationProParams,
    MFOutArgs,
    TargetEstimationCfgParams,
)
from hardtarget.plotting import mf_analysis
from hardtarget.plotting.load_data import load_analysed_data, load_optimized_data
from hardtarget.types.types import ExpParams


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds mandatory and optional positional arguments to the parser."""

    parser.add_argument("path", help="path to source directory with MF output data")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--snr_dB_limit", type=float, default=15.0)
    parser.add_argument("--not_monostatic", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    """Plot MF CLI"""
    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)
    if args.chunk_size == 0:
        args.chunk_size = None

    data_generator: Generator[
        tuple[MFOutArgs, ExpParams, TargetEstimationCfgParams, ExtendedTargetEstimationProParams], None, None
    ] = load_analysed_data(
        args.path,
        start_time=args.start_time,
        end_time=args.end_time,
        relative_time=args.relative_time,
        chunk_size=args.chunk_size,
    )

    optimized_data = load_optimized_data(args.path)

    for out_data, exp_params, cfg_params, pro_params in data_generator:
        fig, axes = plt.subplots(2, 2)
        mf_analysis.plot_peaks(
            axes,
            out_data,
            exp_params,
            cfg_params,
            pro_params,
            optimized_data,
            monostatic=not args.not_monostatic,
            snr_dB_limit=args.snr_dB_limit,
        )

        fig, axes = plt.subplots(2, 3)
        mf_analysis.plot_detections(
            axes,
            out_data,
            exp_params,
            cfg_params,
            pro_params,
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
        mf_analysis.plot_map(
            axes,
            out_data,
            exp_params,
            cfg_params,
            pro_params,
        )

        plt.show()
