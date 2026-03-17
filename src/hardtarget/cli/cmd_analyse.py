"""
The CLI Analyse functionality, abstracts the analyse functionality to a user friendly CLI interface
"""

import argparse
import logging

from radardef import RadarDef

import hardtarget.utils.global_mpi
from hardtarget.analyse import analyse
from hardtarget.constants import (
    AnalysisMethod,
    DOAMethod,
    EventDetectionMethod,
    Impl,
    OptimizationMethod,
    StrEnum,
    TargetEstimationMethod,
)
from hardtarget.types import Array, ArrayKwargs, ArrayParams, Job
from hardtarget.utils.profiling import get_logging_level

from .commands import add_command


class AnalyseParser:
    logger = logging.getLogger(__name__)

    def __init__(self, method_name: AnalysisMethod, sub_methods: type[StrEnum]) -> None:
        self.method_name = method_name
        self.sub_methods = sub_methods

    def parser_build(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds mandatory and optional positional arguments to the parser."""

        parser.add_argument("data", help="path to source directory with rx data")
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
        parser.add_argument("-s", "--start_time", default=None, type=int)
        parser.add_argument("-e", "--end_time", default=None, type=int)
        parser.add_argument("--relative_time", action="store_true")
        parser.add_argument("--clobber", action="store_true", help="override outputs")
        parser.add_argument(
            "-i",
            "--implementation",
            choices=[impl.value for impl in Impl],
            help="implementation",
            default=None,
        )
        if self.method_name is AnalysisMethod.direction_of_arrival:
            parser.add_argument(
                "station_id",
                help="Source of measurement",
                choices=[f"{station_id}" for station_id in RadarDef().radar_stations],
            )

        return parser

    def main(self, args: argparse.Namespace) -> None:
        """Analyse CLI"""

        # Logging
        self.logger.setLevel(get_logging_level(args.verbose))

        # import mpi (in case script is run by mpi)
        comm = hardtarget.utils.global_mpi.import_mpi()

        # job
        job = Job(idx=comm.rank, N=comm.size)

        if self.method_name == AnalysisMethod.direction_of_arrival:
            # TODO: How to handle inputs to the radar_station (just default now)
            radar_station = RadarDef().get_radar(args.station_id)
            if radar_station is None:
                raise KeyError(f"No station available with id: {args.station_id}")
            if not isinstance(radar_station.beam, Array) or not isinstance(
                radar_station.beam_parameters, ArrayParams
            ):
                raise ValueError(
                    f"Radar station {radar_station.station_id} is not of Array type, not possible to calculate direction of arrival"
                )
            kwargs = ArrayKwargs(beam=radar_station.beam, parameters=radar_station.beam_parameters)
        else:
            kwargs = {}

        # process
        results = analyse(
            path=args.data,
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
            **kwargs,
        )

        self.logger.info(f"produced {len(results['files'])} files")


method_and_sub_method = [
    (AnalysisMethod.target_estimation, TargetEstimationMethod),
    (AnalysisMethod.optimize, OptimizationMethod),
    (AnalysisMethod.event_detection, EventDetectionMethod),
    (AnalysisMethod.direction_of_arrival, DOAMethod),
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
