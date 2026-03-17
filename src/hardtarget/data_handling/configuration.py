"""
Main module for configuring the analysis of the input data and calculating the
relevant parameters needed for analysis.
"""

# TODO: the current analysis might not handle partial codes, add this functionality

import configparser
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
import numpy.typing as npt

from hardtarget.constants import AnalysisMethod, ConfigSubSection, MethodLib
from hardtarget.types import CfgParams, ExpParams, Impl, ProParams


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


def extract_config_params_from_derived_object(cfg_derived: CfgParams) -> CfgParams:
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

    # ---- signal indexing ----
    rx_stencil = np.full((read_length,), False, dtype=bool)
    tx_stencil = np.full((read_length,), False, dtype=bool)

    # ---- start and end samples ----
    rx_start_samp = usec_to_samp(exp_params.t_rx_start_usec)
    rx_end_samp = usec_to_samp(exp_params.t_rx_end_usec)
    tx_start_samp = usec_to_samp(exp_params.t_tx_start_usec)
    tx_end_samp = usec_to_samp(exp_params.t_tx_end_usec)
    tx_pulse_samps = tx_end_samp - tx_start_samp

    # ---- Range gates ----
    _il0_rgs_min = tx_start_samp + 1
    _il0_rgs_max = rx_end_samp - tx_pulse_samps
    _il0_min_range_gate = cfg_params.min_range_gate
    _il0_min_range_gate += _il0_rgs_min if cfg_params.min_range_gate >= 0 else _il0_rgs_min
    _il0_max_range_gate = cfg_params.max_range_gate
    _il0_max_range_gate += _il0_rgs_min if cfg_params.max_range_gate >= 0 else _il0_rgs_max

    min_range_gate = _il0_min_range_gate - _il0_rgs_min
    max_range_gate = _il0_max_range_gate - _il0_rgs_min

    range_gates = np.arange(
        min_range_gate,
        max_range_gate,
        cfg_params.range_gate_step,
        dtype=np.int32,
    )
    rel_rgs = (range_gates - cfg_params.min_range_gate).astype(np.int32)

    # --- Signal stencils ----
    for k in range(cfg_params.n_ipp):
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
