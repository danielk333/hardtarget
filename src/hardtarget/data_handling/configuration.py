"""
Main module for configuring the analysis of the input data and calculating the
relevant parameters needed for analysis.

The analysis always assumes the radar data consists of a
continuous stream of signal samples. If in reality there was no samples taken
during an interval the sample stream should be zero-padded during this time.
As such the signal sample stream is split up into regular cycles. Each cycle
occurs back-to-back and the time a cycle takes is called an inter-pulse-period,
or IPP.

Every analysis consists of one or several received signals (RX) and can also contain a
transmitted signal (TX). If no transmitted signal is available it should be simulated.
These signals can either be superimposed in the same channel or be in different
channels in the data structure. Either way, there are several segments within
each cycle that we need to extract and index. Hence, there are three main levels
of signal indices within one cycle, which we will call Index Levels (IL's):

```
    0) Signal samples (can be same channel)
        - RX signal (size=IPP length)
        - TX signal (size=IPP length)
    0d) Decimated signal samples
        - RX signal (size=IPP length / decimation)
    1) Stenciled samples:
        - RX window (size=chosen reception length, i.e. all range gates)
        - TX pulse (size=length of transmitted pulse)
    2) Target range-gate:
        - RX pulse (size=length of transmitted pulse, offset by chosen range gate)
```
The IL-0d is a bit of a special case as temporal decimation (piece-wise sums of
neighbors) only maintains desirable properties (both statistical and signal
vise) if it is done on an isochronal and continuous sample stream. As such,
decimation operation only occurs on a level 0 signal.

Illustration of the above:

```

    IL-0     : |0123456789...................................|
    Signal   : |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|
    IL-0d    : | 0  1  2  3  4  5  6  7  8  9  .  .  .  .  . |
    Decimated: | x  x  x  x  x  x  x  x  x  x  x  x  x  x  x |
    IL-1 tx  : |  012345                                     |
    TX pulse : |--xxxxxx-------------------------------------|
    IL-1 rx  : |                0123456789..............     |
    RX window: |----------------xxxxxxxxxxxxxxxxxxxxxxxx-----|
    IL-2     : |                           012345            |
    RX pulse : |---------------------------xxxxxx------------|
```
In the above example, the first sample of the TX pulse would have index 2 in
terms of IL-0 but would have index 0 in terms of IL-1. The first sample of IL-0d
would represent samples 0, 1, and 2 in IL-0. Similarly the first sample of the
RX pulse would have index 27 in terms of IL-0, index 11 in IL-1 and index 0 in
IL-2. Also, index 0, 1, and 2 in the RX pulse IL-2 would be associated with only
index 9 in IL-0d.

The general strategy for decimation is to zero-pad the target vector if the
decimation does not evenly divide the vector.

In the analysis, several of these cycles are stacked on top of each other. This
means that when selecting the IL-1 tx samples the resulting array is no longer
isochronal and continues but instead has a jump in the middle, e.g.


```
    IL-1 tx  : |  012345        6789..        ......      |
    TX pulse : |--xxxxxx--------xxxxxx--------xxxxxx------|
```
Since range from a transmitter converted to time is measured from the start of
transmission (i.e. the time between when the leading edge of the wave leaves the
transmitter and reaches the receiver), the range gates are also always measured
in samples relative the start of the TX pulse + 1, in terms of IL-0. Which means
that sample of range-gate 0 has traveled the time of 1 sample. The largest range
gate is at the end of reception, which basically means only one sample of the TX
could be measured.

#TODO: Move explanation to docs aswell and MF process

"""

# TODO: decimation runs on JUST the top 1 level of direct signal
# TODO: the current analysis might not handle partial codes, add this functionality

import configparser
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
import numpy.typing as npt

from hardtarget.types.constants import AnalysisMethod, ConfigSubSection, MethodLib
from hardtarget.types.types import CfgParams, ExpParams, GenericCfg, Impl, ProParams


def extract_config_section(
    cfg_pth: Path,
    section: str,
    cfg_type: Type[CfgParams],
    existing_cfg: Optional[CfgParams] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """
    Extract a specific section from the given .ini file, if a exising config the new section will be
    appended upon it.

    Args:
        cfg_pth: Path to config
        section: Config section to extract data from
        cfg_type: Config type
        existing_cfg (optional): Existing configuration that shold be extended

    Returns:
        Dictionary of key value format of the given config
    """

    if existing_cfg is not None:
        default_dict = asdict(cfg_type(**asdict(existing_cfg)))
    else:
        default_dict = asdict(cfg_type())

    if not cfg_pth.exists():
        raise FileNotFoundError(f"Config file '{cfg_pth}' does not exist")
    if cfg_pth.is_dir():
        raise IsADirectoryError(f"Config file '{cfg_pth}' is a directory")

    config = configparser.ConfigParser()
    config.read(cfg_pth)

    try:
        for key in config[section].keys():
            if key not in default_dict.keys():
                raise ValueError(f"'{key}' option found in config file is not a valid config parameter")
            # Convert values to specific types
            if isinstance(default_dict[key], int):
                default_dict[key] = config.getint(section, key)
            elif isinstance(default_dict[key], bool):
                default_dict[key] = config.getboolean(section, key)
            elif isinstance(default_dict[key], float):
                default_dict[key] = config.getfloat(section, key)
            else:
                # string
                default_dict[key] = config.get(section, key).strip("'").strip('"')
    except KeyError:
        if logger:
            logger.warning(f"No subsection {section} available in config file, default values will be used")

    return default_dict


def load_config_params(configfile: Path | str) -> CfgParams:
    """
    Load config parameters from the processing section.

    Args:
        configfile: Path to configfile

    Returns:
        Dataclass containing the configuration information.

    """

    cfg_pth = Path(configfile)
    d = extract_config_section(cfg_pth, ConfigSubSection.PROCCESSING, CfgParams)
    return CfgParams(**d)


def extract_config_params_from_derived_object(cfg_derived: GenericCfg) -> CfgParams:
    """
    From a derived class extract the fields from the base class, this to extract the non process related
    parameters.

    Args:
        cfg_derived: Any class that inherits from CfgParams

    Returns:
        Configuration parameters only relevant to the base class
    """

    derived = asdict(cfg_derived)

    base = {}
    for key in fields(CfgParams):
        base[key.name] = derived[key.name]

    return CfgParams(**base)


def compute_process_params(
    exp_params: ExpParams,
    cfg_params: CfgParams,
    analysis_method: AnalysisMethod,
    method_lib: Optional[MethodLib] = None,
    implementation: Optional[Impl] = None,
) -> ProParams:
    """
    Computes the processing parameters from the experiment and configuration parameters.

    Args:
        exp_params: Experiment parameters from dataloader
        cfg_params: Configuration parameters from user
        analysis_method: Method to use during analysis
        method_lib: Specific library used

    Returns:
        Process parameters derived from the given input
    """

    def usec_to_samp(usec: int | float) -> int:
        return int(usec / exp_params.t_samp_usec)

    # ---- Read length ----
    read_length = (cfg_params.n_ipp + cfg_params.ipp_offset) * exp_params.ipp_samps
    # ---- Range gates ----
    range_gates = np.arange(
        cfg_params.min_range_gate, cfg_params.max_range_gate, cfg_params.range_gate_step, dtype=np.int32
    )
    rel_rgs = (range_gates - cfg_params.min_range_gate).astype(np.int32)

    # --- signal indexing ----
    rx_stencil = np.full((read_length,), False, dtype=bool)
    tx_stencil = np.full((read_length,), False, dtype=bool)

    # --- Signal stencils ----
    for k in range(cfg_params.n_ipp):
        rx_end_samp = int(exp_params.t_rx_end_usec / exp_params.t_samp_usec)
        tx_start_samp = int(exp_params.t_tx_start_usec / exp_params.t_samp_usec)
        tx_end_samp = int(exp_params.t_tx_end_usec / exp_params.t_samp_usec)
        tx_pulse_samps = tx_end_samp - tx_start_samp

        _il0_min_range_gate = cfg_params.min_range_gate
        # TODO: should it be else rx_start_samp??
        _il0_min_range_gate += (tx_start_samp + 1) if cfg_params.min_range_gate >= 0 else (tx_start_samp + 1)
        _il0_max_range_gate = cfg_params.max_range_gate
        _il0_max_range_gate += (
            (tx_start_samp + 1) if cfg_params.max_range_gate >= 0 else (rx_end_samp - tx_pulse_samps)
        )

        # start of pulse within range-gates, thus include also the entire pulse at the end
        _rx0 = (k + cfg_params.ipp_offset) * exp_params.ipp_samps + _il0_min_range_gate
        _rx1 = (k + cfg_params.ipp_offset) * exp_params.ipp_samps + _il0_max_range_gate + tx_pulse_samps
        rx_stencil[_rx0:_rx1] = True

        _tx0 = k * exp_params.ipp_samps + tx_start_samp
        _tx1 = k * exp_params.ipp_samps + tx_end_samp
        tx_stencil[_tx0:_tx1] = True

    return ProParams(
        method=analysis_method,
        method_lib=method_lib,
        implementation=implementation,
        read_length=read_length,
        range_gates=range_gates,
        rel_rgs=rel_rgs,
        rx_stencil=rx_stencil,
        tx_stencil=tx_stencil,
    )


def get_ilx_windows(
    exp_params: ExpParams, cfg_params: CfgParams, pro_params: ProParams
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    rx_end_samp = int(exp_params.t_rx_end_usec / exp_params.t_samp_usec)
    tx_start_samp = int(exp_params.t_tx_start_usec / exp_params.t_samp_usec)
    tx_end_samp = int(exp_params.t_tx_end_usec / exp_params.t_samp_usec)

    _tx_pulse_samps = tx_end_samp - tx_start_samp

    # range gates are relative to tx start + 1
    # TODO: this is one of the parts of not handling partial codes
    _il0_rgs_min = tx_start_samp + 1
    _il0_rgs_max = rx_end_samp - _tx_pulse_samps
    _il0_max_range_gate = cfg_params.max_range_gate
    _il0_max_range_gate += _il0_rgs_min if cfg_params.max_range_gate >= 0 else _il0_rgs_max
    _il0_min_range_gate = cfg_params.min_range_gate
    _il0_min_range_gate += _il0_rgs_min if cfg_params.min_range_gate >= 0 else _il0_rgs_min

    il0_rgs = np.arange(_il0_min_range_gate, _il0_max_range_gate, cfg_params.range_gate_step, dtype=np.int32)

    _il0_rx_stencil_indices = np.argwhere(pro_params.rx_stencil).flatten()

    # cyclic range gate selector - in the index space of stenciled RX signals
    # TODO: this can be generalized for a-periodic codes ect
    _base_rx_window = np.arange(_tx_pulse_samps, dtype=np.int32)
    _d_window = _tx_pulse_samps + len(pro_params.range_gates)
    _il1_rx_window_blocks = [_base_rx_window + ind * _d_window for ind in range(cfg_params.n_ipp)]
    il1_rx_window_indices = np.concatenate(_il1_rx_window_blocks, dtype=np.int32)
    il0_rx_window_indices = _il0_rx_stencil_indices[il1_rx_window_indices].astype(np.int32)

    return il0_rgs, il0_rx_window_indices, il1_rx_window_indices
