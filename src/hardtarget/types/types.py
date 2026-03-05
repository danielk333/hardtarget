"""
This module contains convenient type information so that typing can be precise
but not too verbose in the code itself.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, NamedTuple, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from radardef.types import ExpParams, Pointing

from hardtarget.types.constants import AnalysisMethod, EstimationMethod, Impl

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    # in python 3.10 there is a bug present for TypedDict
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


@dataclass(frozen=True)
class CfgParams:
    """
    Configuration parameters

    Args:
        implementation: Implementation of the library that should be used, very optional.
        method: What method to use for the analysis.
        node_gpus: Amount of gpus
        n_ipp: Number or interpulse periods to coherently integrate
        ipp_offset: Pulse offset to use when selecting transmitt pulse for coherent integration,
                    used for finding range-aliased echoes. Value of 1 would mean first range alias,
                    2 second alias, ect.
        samp_offset: Sample offset, if the measurement does not start at ipp sample 0 an offset can be added.
        min_range_gate: Minimum range gate to process relative to the start of transmission
                        for each pulse (each range gate is one receiver sample).
        max_range_gate: Maximum range gate to process relative to the start of transmission
                        for each pulse (each range gate is one receiver sample).
        range_gate_step: The step in range-gates to use when processing, if baud-length is longer
                         than the receiver sampling time this can be increased to sacrifice range-resolution
                         for processing speed.
        range_gate_sub_resolution: For increased accuracy subresolution can be used, increases resolution by
                                   generating sub samples between then available samples. This will increase computational time.
        frequency_decimation: How many samples to step in range in the first stage of the course search.
        clutter_length: How many samples to remove due to ground clutter
        min_acceleration: Minimum acceleration to search for (m/s^2)
        max_acceleration: Maximum acceleration to search for (m/s^2)
        num_cohints_per_file: How many coherent integration periods to include in one output file.
                              Smaller means that lower latency can be achieved.
        optimization: If optimization should be used.
        tx_amp_limit: The tx amplitude limit, if lower than this the analysis will ignore the cohints.
    """

    implementation: Impl = Impl.numpy
    method: EstimationMethod = EstimationMethod.fgmf
    node_gpus: int = 1
    n_ipp: int = 10
    ipp_offset: int = 0
    samp_offset: int = 0
    min_range_gate: int = 0
    max_range_gate: int = -1
    range_gate_step: int = 1
    range_gate_sub_resolution: int = 1
    frequency_decimation: int = 1
    clutter_length: int = 0
    min_acceleration: float = -200.0
    max_acceleration: float = 200.0
    num_cohints_per_file: int = 100
    optimization: bool = False
    tx_amp_limit: float = 1.0


@dataclass(frozen=True)
class ProParams:
    """
    Process parameters, derived from Cfg and Exp params

    Args:
        method: Estimation method used for the analysis.
        read_length: How many samples should be read each "loop".
        decimated_read_length: Decimated read length, the read length with the frequency decimation accounted
                               for.
        range_gates: Range gates, the "gates" between max and min range gate with a range gate step.
        rel_rgs: Relative range gates, range gates relative to the min range gate.
        il0_rgs: Index level 0 range gates.
        ranges: Range gates (including subresolution) in meter, true ranges.
        rx_stencil: Bool stencil the size of an IPP, each sample containing a rx samples is declare True.
        tx_stencil: Bool stencil the size of an IPP, each sample containing a tx samples is declare True.
        il1_rx_window_indices: Index level 1 receiver window indices.
        il0_rx_window_indices: Index level 0 receiver window indices.
        il0_dec_rx_window_indices: Index level 0 decimated receiver window indices
                                   (il0_rx_window_indices with frequency decimation).
        range_rates: Range rates
    """

    method: AnalysisMethod = AnalysisMethod.unknown
    read_length: int = 0
    decimated_read_length: int = 0
    range_gates: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    rel_rgs: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    il0_rgs: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    ranges: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))
    rx_stencil: npt.NDArray = field(default_factory=lambda: np.ndarray([0, 0], dtype=bool))
    tx_stencil: npt.NDArray = field(default_factory=lambda: np.ndarray([0, 0], dtype=bool))
    il1_rx_window_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    il0_rx_window_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    il0_dec_rx_window_indices: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(2, dtype=np.int32)
    )
    range_rates: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))


class ExtractedSignals(NamedTuple):
    """
    Collection of signals read from a measurement file.
    Args:
        z_tx: tx signal samples
        z_rx: rx signal samples
        z_ipp: The full ipp samples
    """

    z_tx: npt.NDArray[np.complexfloating]
    z_rx: npt.NDArray[np.complexfloating]
    z_ipp: npt.NDArray[np.complexfloating]


class Bounds(NamedTuple):
    """Bounds, contains start and stop of any arbitrary object"""

    start: int
    end: int


class ParserArgs(TypedDict):
    description: str
    usage: str


class SubParser(NamedTuple):
    main: Callable[[argparse.Namespace], None]
    parser_build: Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
    parser_args: ParserArgs


class DataItem(NamedTuple):
    """
    To save the output data in a unified h5 format each object should be defined as a
    DataItem.

    Args:
        data: Dataset to be stored
        long_name: Longer description of the data
        dims: dataset dimensions
        units: units
        scale: boolean to define if it is a scalar number, if true dims can be left empty
    """

    data: Any
    long_name: str
    dims: list[tuple[str, str]] | None = None
    units: str | None = None
    scale: bool | None = None


class Job(NamedTuple):
    idx: int
    N: int


class OptStart(NamedTuple):
    r_vec: float = 0.0
    v_vec: float = 0.0
    a_vec: float = 0.0


GenericCfg = TypeVar("GenericCfg", bound=CfgParams)
GenericPro = TypeVar("GenericPro", bound=ProParams)
GenericVars = TypeVar("GenericVars", bound=NamedTuple)
GenericOut = TypeVar("GenericOut", bound=NamedTuple)


class AnalysedResult(TypedDict, Generic[GenericOut, GenericCfg, GenericPro]):
    """
    Analysed result should contain the output from the analysis so that it is posisble
    to access the data easily.

    Args:
        dir: directory of stored files
        files: list of all files
        data: if the data is not saved to file the data is stored here during runtime with start sample
              index as key
    """

    dir: str | Path | None
    files: list[str]
    data: dict[int, tuple[GenericOut, ExpParams, GenericCfg, GenericPro]]


AnalysisLib: TypeAlias = Callable[
    [
        npt.NDArray[np.complexfloating],
        npt.NDArray[np.complexfloating],
        npt.NDArray,
        GenericCfg,
        GenericPro,
    ],
    GenericVars,
]

OptimizeLib: TypeAlias = Callable[
    [
        npt.NDArray[np.complexfloating],
        npt.NDArray,
        ExpParams,
        GenericCfg,
        GenericPro,
        OptStart,
    ],
    tuple[npt.NDArray, npt.NDArray],
]
