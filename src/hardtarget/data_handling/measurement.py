"""
The measurement class is used to simplify the usage of raw data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from radardef import DataLoader, RadarDef
from radardef.types import ExpParams, Pointing

from hardtarget.data_handling.configuration import (
    compute_process_params,
    extract_config_params_from_derived_object,
    load_config_params,
)
from hardtarget.data_simulation.tx_model import tx_signal_model
from hardtarget.matched_filter import get_available_libs, get_estimation_method
from hardtarget.types.constants import AnalysisMethod, EstimationMethod
from hardtarget.types.types import (
    AnalysisLib,
    Bounds,
    CfgParams,
    ExtractedSignals,
    GenericCfg,
    Impl,
    OptimizeLib,
    ProParams,
)


class Measurement:
    """

    Wrapper to simplify the usage of the measurement data and user defined configuration.

    Args:
        path: Path to measurement file or directory
        config: Path to config file
        rx_channel (optional): If a specific channel should be analysed, if None all channels will be summed
        excluded_channels (optional): Channels to ignore when if summing over several channels,
                                        thus only applicable when rx_channel is None
        impl (optional): Implementation to be used during analyse (C/Cuda/Numpy)
        est_method (optional): Estimation method to be used during analyse
    """

    @property
    def path(self) -> Path:
        """Path to measurement file"""
        return self.__path

    @property
    def exp_params(self) -> ExpParams:
        """Experiment paramters from the measurement file"""
        return self.__exp_params

    @property
    def cfg_params(self) -> CfgParams:
        """Analyse configuration parameters"""
        return self.__cfg_params

    @property
    def pro_params(self) -> ProParams:
        """Process paramters computed from the experiment and configuration parameters"""
        return self.__pro_params

    @property
    def rx_sample_bounds(self) -> Bounds:
        """Sample bounds (start/stop) of the rx channel/s"""

        start, end = self.__data_loader.bounds(
            self._rx_channel if self._rx_channel is not None else self.data_loader.channels[0]
        )
        return Bounds(start=start, end=end)

    @property
    def tx_sample_bounds(self) -> Bounds:
        """Sample bounds (start/stop) of the tx channel"""
        start, end = self.__data_loader.bounds(str(self.exp_params.tx_channel))
        return Bounds(start=start, end=end)

    @property
    def epoch(self) -> Bounds:
        """Start and stop in microseconds"""
        t_start_usec, t_end_usec = self.data_loader.meta.bounds
        return Bounds(int(t_start_usec), int(t_end_usec))

    @property
    def estimation_method(self) -> AnalysisLib | OptimizeLib:
        """Library to be used for the analyse"""
        return self.__estimation_method

    @property
    def analysis_method(self) -> AnalysisMethod:
        """Analyse method"""
        return self.__analysis_method

    @property
    def data_loader(self) -> DataLoader:
        """Data loader to access the measurment data"""
        return self.__data_loader

    def __init__(
        self,
        path: Path | str,
        config: Path | str | GenericCfg,
        rx_channel: Optional[str | int] = None,
        excluded_channels: Optional[list[str] | list[int]] = None,
        impl: Optional[Impl] = None,
        est_method: Optional[EstimationMethod] = None,
    ) -> None:

        # Access measurement data
        self.__path = Path(path)
        data_loader = RadarDef().load_data(self.__path)
        if data_loader is None:
            raise Exception(f"Not possible to load data file from: {self.__path}")
        else:
            self.__data_loader = data_loader
        self.__exp_params = self.data_loader.meta.experiment
        self._rx_channel, self._tx_channel = self.extract_channels(self.exp_params, rx_channel)
        self._excluded_channels = excluded_channels if excluded_channels is not None else []

        # Extract user config, .ini file or CfgParams type object
        if isinstance(config, CfgParams):
            self.__cfg_params = extract_config_params_from_derived_object(
                config, impl=impl, est_method=est_method
            )
        else:
            self.__cfg_params = load_config_params(config, impl=impl, est_method=est_method)

        # Extract estimation method and anlysis method to run
        estimation_method, self.__analysis_method = get_estimation_method(
            self.cfg_params.implementation,
            self.cfg_params.method,
        )
        if estimation_method is None:
            raise ValueError(
                f"Cannot find requested method '{self.cfg_params.method}' "
                f"in requested implementation '{self.cfg_params.implementation}'\n"
                f"Available implemented methods: \n{get_available_libs(indent=' ' * 4)}"
            )
        else:
            self.__estimation_method = estimation_method

        # Derive process parameters
        self.__pro_params = compute_process_params(self.exp_params, self.cfg_params, self.__analysis_method)

    def extract_channels(
        self, exp_params: ExpParams, rx_channel: Optional[int | str] = None
    ) -> tuple[int | str | None, int | str | None]:
        """
        From experiment parameters and requested rx channel extract the correct ones from the data file
        """

        _tx_channel = exp_params.tx_channel
        _rx_channel = None
        if rx_channel is not None:
            _rx_channel = rx_channel
            if _rx_channel not in exp_params.rx_channels:
                raise ValueError(f"rx_channel: {rx_channel} is not a valid channel in the measurementfile")
        elif len(exp_params.rx_channels) == 1:
            # Only one available rx_channel, thus we can declare it here
            _rx_channel = exp_params.rx_channels[0]

        return _rx_channel, _tx_channel

    def pointing(self, start_sample: int) -> Pointing:
        """Pointing data to define radar pointing direction in spherical coordinates"""
        return self.__data_loader.pointing(start_sample)

    def extract_signals(self, start_sample: int, read_length: int) -> ExtractedSignals:
        """
        Extract rx and tx data at the given sample.

        Args:
            start_sample: Start sample
            read_length: Amount of sample to read from the start sample

        Returns:
            Rx and Tx samples, If multiple rx channels are available and no specific channel has been
            requested during initialization all channels except the tx channel and channels in the excluded
            channels list will be summed. If no tx channel is available a tx model
            will be used to simulate the tx signal
        """

        # if no rx channel specified all channels will be summed for full analysis
        if self._rx_channel is None:
            z_ipp = np.zeros((read_length,), dtype=np.complex128)
            for chnl in self.data_loader.channels:
                if chnl != self._tx_channel and chnl not in self._excluded_channels:
                    z_ipp += self.data_loader.read(chnl, start_sample, read_length)
        else:
            z_ipp = self.data_loader.read(self._rx_channel, start_sample, read_length)

        # Extracting tx data
        if self._tx_channel != self._rx_channel and self._tx_channel is not None:
            z_tx = self.data_loader.read(self._tx_channel, start_sample, read_length)
            z_tx = np.broadcast_to(
                z_tx.reshape((z_tx.size, 1)), (z_tx.size, self.cfg_params.range_gate_sub_resolution)
            )
        elif self._tx_channel == self._rx_channel and self._tx_channel is not None:
            z_tx = z_ipp.copy()
            z_tx = np.broadcast_to(
                z_tx.reshape((z_tx.size, 1)), (z_tx.size, self.cfg_params.range_gate_sub_resolution)
            )
        else:
            assert self.exp_params.code is not None, (
                "No code available from the metadata, not possible to simulate tx"
            )
            z_tx = tx_signal_model(
                code=self.exp_params.code,
                ipp_samps=self.exp_params.ipp_samps,
                read_length=read_length,
                start_sample=start_sample,
                sub_resolution=self.cfg_params.range_gate_sub_resolution,
                kind="linear",
            )

        # clean ground clutter, get separate transmit waveform and echo vectors
        z_rx = z_ipp[self.pro_params.rx_stencil].copy()
        z_tx = z_tx[self.pro_params.tx_stencil, :]
        return ExtractedSignals(z_tx=z_tx, z_rx=z_rx, z_ipp=z_ipp)
