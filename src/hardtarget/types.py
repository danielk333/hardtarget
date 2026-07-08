"""
This module contains convenient type information so that typing can be precise
but not too verbose in the code itself.
"""

import argparse
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, NamedTuple, Protocol, Self, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from pyant.models.array import Array, ArrayParams
from radardef.types import ExpDef

from hardtarget.constants import (
    AnalysisMethod,
    Impl,
    MethodLib,
)

if (sys.version_info.major, sys.version_info.minor) <= (3, 10):
    # in python 3.10 there is a bug present for TypedDict
    from typing_extensions import Self, TypedDict
else:
    from typing import Self, TypedDict


@dataclass(frozen=True)
class CfgParams:
    """
    Configuration parameters

    Args:
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
        num_cohints_per_file: How many coherent integration periods to include in one output file.
                              Smaller means that lower latency can be achieved.
        tx_amp_limit: The tx amplitude limit, if lower than this the analysis will ignore the cohints.
        node_gpus: Amount of gpus
    """

    n_ipp: int = 1
    ipp_offset: int = 0
    samp_offset: int = 0
    min_range_gate: int = 0
    max_range_gate: int = -1
    range_gate_step: int = 1
    num_cohints_per_file: int = 100
    tx_amp_limit: float = 1.0
    node_gpus: int = 1
    cache: bool = True


@dataclass(frozen=True)
class ProParams:
    """
    Process parameters, derived from Cfg and Exp params

    Args:
        method: Method used for the analysis.
        method_lib: Specific library used for the analyse method.
        implementation: Implementation of the method C/Cuda/Numpy
        read_length: How many samples to read per
        rx_stencil: Bool stencil the size of an IPP, each sample containing a rx samples is declare True.
        tx_stencil: Bool stencil the size of an IPP, each sample containing a tx samples is declare True.
        range_gates: Range gates, the "gates" between max and min range gate with a range gate step.
        rel_rgs: Relative range gates, range gates relative to the min range gate.
    """

    method: AnalysisMethod = AnalysisMethod.unknown
    method_lib: MethodLib | None = None
    implementation: Impl | None = None
    read_length: int = 1
    rx_stencil: npt.NDArray = field(default_factory=lambda: np.zeros((1,), dtype=bool))
    tx_stencil: npt.NDArray = field(default_factory=lambda: np.zeros((1,), dtype=bool))
    rel_rgs: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    range_gates: npt.NDArray[np.int32] = field(default_factory=lambda: np.zeros((1,), dtype=np.int32))


@dataclass(frozen=True)
class OutputBase:
    """
    Base output need for any process

    epoch_us: Date of first analysed sample
    """

    epoch_us: int

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Used to add each variable name to annotations

        """
        super().__init_subclass__(**kwargs)
        for field_name in cls.__annotations__:
            setattr(cls, field_name, field_name)

    def copy_and_concatenate(self, additonal_data: Self) -> Self:
        concatenated_data = asdict(self)
        for arg in fields(additonal_data):
            key = arg.name
            data = getattr(additonal_data, key)
            # only interested in the epoch start of the measurement TODO: adjust this
            if key == f"{self.epoch_us=}".split("=")[0].split(".")[1]:
                continue
            if isinstance(data, np.ndarray):
                if data.ndim >= 2:
                    concatenated_data[key] = np.vstack([concatenated_data[key], data])
                else:
                    concatenated_data[key] = np.hstack([concatenated_data[key], data])
                # concatenated_data[key] = np.append(concatenated_data[key], data, axis=0)
            else:
                concatenated_data[key] = concatenated_data[key] + data

        return self.__class__(**concatenated_data)


class ExtractedSignals(NamedTuple):
    """
    Collection of signals read from a measurement file.
    Args:
        tx: tx signal samples
        rx: rx signal samples
        ipp: The full ipp samples
    """

    tx: npt.NDArray[np.complexfloating]
    rx: npt.NDArray[np.complexfloating]
    ipp: npt.NDArray[np.complexfloating]


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


class ArrayKwargs(TypedDict, total=False):
    beam: Array
    parameters: ArrayParams


GenericCfg = TypeVar("GenericCfg", bound=CfgParams)
GenericPro = TypeVar("GenericPro", bound=ProParams)
GenericVars = TypeVar("GenericVars", bound=NamedTuple)
GenericOut = TypeVar("GenericOut", bound=OutputBase)
GenericLib = TypeVar("GenericLib", bound=Callable)


class AnalysedResult(TypedDict, Generic[GenericOut, GenericCfg, GenericPro]):
    """
    Analysed result should contain the output from the analysis so that it is posisble
    to access the data easily.

    Args:
        dir: output directory of stored files
        file_dir: List of each root directory of the files.
        files: list of all files
        data: if the data is not saved to file the data is stored here during runtime with start sample
              index as key
    """

    dir: str | Path | None
    file_dir: list[Path]
    files: list[str]
    data: dict[int, tuple[GenericOut, ExpDef, GenericCfg, GenericPro]]


TargetEstimationLib: TypeAlias = Callable[
    [
        npt.NDArray[np.complex64],
        npt.NDArray[np.complex64],
        npt.NDArray[np.floating],
        ExpDef,
        GenericCfg,
        GenericPro,
    ],
    GenericVars,
]

OptimizeLib: TypeAlias = Callable[
    [npt.NDArray[np.complex64], npt.NDArray, ExpDef, GenericCfg, GenericPro, float, float, float],
    tuple[float, float, float, float],
]

EchoSearchLib: TypeAlias = Callable[
    [
        npt.NDArray[np.complex64],
        npt.NDArray[np.complex64],
        ExpDef,
        GenericCfg,
        GenericPro,
    ],
    GenericVars,
]

InterferometryLib: TypeAlias = Callable[
    [npt.NDArray[np.complex64], ExpDef, GenericCfg, GenericPro, Array, ArrayParams],
    GenericVars,
]

LibType: TypeAlias = TargetEstimationLib | OptimizeLib | EchoSearchLib | InterferometryLib


# Type hinting for a common declaration of what func_get_data should be passed to the processes
class ExtractSignals(Protocol):
    def __call__(
        self,
        start_sample: int,
        read_length: int,
        sum_rx_channels: bool = True,
        sub_resolution: int = 1,
    ) -> ExtractedSignals: ...


# Protocol to be able specify DataClass input
class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]
