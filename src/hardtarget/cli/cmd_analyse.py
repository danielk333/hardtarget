"""
The CLI Analyse functionality, abstracts the analyse functionality to a user friendly CLI interface
"""

import argparse
import logging

import hardtarget.utils.global_mpi
from hardtarget.analyse import analyse
from hardtarget.types.constants import (
    AnalysisMethod,
    EventDetectionMethod,
    Impl,
    OptimizationMethod,
    StrEnum,
    TargetEstimationMethod,
)
from hardtarget.types.types import Job
from hardtarget.utils.profiling import get_logging_level

from .commands import add_command


class AnalyseParser:
    logger = logging.getLogger(__name__)

    def __init__(self, method_name: AnalysisMethod, sub_methods: type[StrEnum]) -> None:
        self.method_name = method_name
        self.sub_methods = sub_methods

    def parser_build(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds mandatory and optional positional arguments to the parser."""

        parser.add_argument("rx", help="path to source directory with rx data")
        parser.add_argument(
            "-m",
            "--method",
            help=f"{self.method_name} method",
            choices=[f"{method}" for method in self.sub_methods],
            default=None,
        )
        parser.add_argument("--rxchnl", help="RX channel")
        parser.add_argument("--config", help="path to config file for GMF processing")
        parser.add_argument("-o", "--output", default=".", help="path to output directory")
        parser.add_argument("-p", "--progress", action="store_true", help="enable progress bar")
        parser.add_argument("-s", "--start_time", default=None)
        parser.add_argument("-e", "--end_time", default=None)
        parser.add_argument("--relative_time", action="store_true")
        parser.add_argument("--clobber", action="store_true", help="override outputs")
        parser.add_argument(
            "-i",
            "--implementation",
            choices=[impl.value for impl in Impl],
            help="implementation",
            default=None,
        )
        return parser

    def main(self, args: argparse.Namespace) -> None:
        """Analyse CLI"""

        # Logging
        self.logger.setLevel(get_logging_level(args.verbose))

        if args.relative_time:
            args.start_time = float(args.start_time)
            args.end_time = float(args.end_time)

        # import mpi (in case script is run by mpi)
        comm = hardtarget.utils.global_mpi.import_mpi()

        # job
        job = Job(idx=comm.rank, N=comm.size)

        # process
        results = analyse(
            path=args.rx,
            rx_channel=args.rxchnl,
            config=args.config,
            job=job,
            method=self.method_name,
            method_lib=args.method,
            implementation=args.implementation,
            clobber=args.clobber,
            output=args.output,
            start_time=args.start_time,
            end_time=args.end_time,
            relative_time=args.relative_time,
            progress=args.progress,
        )

        self.logger.info(f"produced {len(results['files'])} files")


method_and_sub_method = [
    (AnalysisMethod.target_estimation, TargetEstimationMethod),
    (AnalysisMethod.optimize, OptimizationMethod),
    (AnalysisMethod.event_detection, EventDetectionMethod),
]

for method, sub_method in method_and_sub_method:
    parser = AnalyseParser(method, sub_method)

    add_command(
        name=method,
        function=parser.main,
        parser_build=parser.parser_build,
        add_parser_args=dict(
            description=f"Script for running {method} analysis",
        ),
    )
