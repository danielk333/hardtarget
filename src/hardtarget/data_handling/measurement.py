"""
The measurement class is used to simplify the usage of raw data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from radardef import DataLoader, RadarDef
from radardef.types import ExpDef, Pointing

from hardtarget.constants import AnalysisMethod, MethodLib
from hardtarget.data_handling.configuration import (
    compute_process_params,
    extract_config_from_dict,
    extract_config_params_from_derived_object,
    load_config_params,
)
from hardtarget.data_simulation.tx_model import tx_signal_model
from hardtarget.types import (
    Bounds,
    CfgParams,
    ExtractedSignals,
    Impl,
    ProParams,
)


class Measurement:
    """

    Wrapper to simplify the usage of the measurement data and user defined configuration.

    Args:
        path: path to the measurement data
        config: Path to config file
        method: Analysis method to be used
        method_lib (optional): Specific method library
        impl (optional): Implementation to be used during analyse (C/Cuda/Numpy)
        rx_channel (optional): If a specific channel should be analysed, if None all channels will be summed
        excluded_channels (optional): Channels to ignore when if summing over several channels,
                                        thus only applicable when rx_channel is None
    """

    @property
    def exp_params(self) -> ExpDef:
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
        t_start_usec, t_end_usec = self.data_loader.epoch_bounds
        return Bounds(int(t_start_usec), int(t_end_usec))

    @property
    def data_loader(self) -> DataLoader:
        """Data loader to access the measurment data"""
        return self.__data_loader

    def __init__(
        self,
        path: Path | str,
        config: Path | str | CfgParams | dict,
        method: AnalysisMethod,
        method_lib: Optional[MethodLib] = None,
        impl: Optional[Impl] = None,
        rx_channel: Optional[str | int] = None,
        excluded_channels: Optional[list[str] | list[int]] = None,
        exp_params: Optional[ExpDef] = None,
    ) -> None:

        # Access measurement data
        self.__path = Path(path)
        data_loader = RadarDef().load_data(self.__path, experiment=exp_params)
        if data_loader is None:
            raise Exception(f"Not possible to load data file from: {self.__path}")
        else:
            self.__data_loader = data_loader
        self.__exp_params = self.data_loader.experiment
        self._rx_channel, self._tx_channel = self.extract_channels(self.exp_params, rx_channel)
        self._excluded_channels = excluded_channels if excluded_channels is not None else []

        # Extract user config, .ini file or CfgParams type object
        if isinstance(config, CfgParams):
            self.__cfg_params = extract_config_params_from_derived_object(config)
        elif isinstance(config, dict):
            self.__cfg_params = extract_config_from_dict(config, CfgParams)
        else:
            self.__cfg_params = load_config_params(config)

        # Derive process parameters
        self.__pro_params = compute_process_params(
            self.exp_params,
            self.cfg_params,
            analysis_method=method,
            method_lib=method_lib,
            implementation=impl,
        )

    def extract_channels(
        self, exp_params: ExpDef, rx_channel: Optional[int | str] = None
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

    def extract_signals(
        self,
        start_sample: int,
        read_length: int,
        sum_rx_channels: bool = True,
        sub_resolution: int = 1,
    ) -> ExtractedSignals:
        """
        Extract rx and tx data at the given sample.

        Args:
            start_sample: Start sample
            read_length: Amount of sample to read from the start sample
            sum_rx_channels (optional): If multiple rx channels available they will all be summed.
        Returns:
            Rx and Tx samples, either as a array (read_length,) or if multiple channels requested
            (n_channels, read_length).
            If no tx channel is available a tx model will be used to simulate the tx signal
        """

        # if no rx channel specified all channels will be summed for full analysis
        if sum_rx_channels:
            if self._rx_channel is None:
                ipp = np.zeros((read_length,), dtype=np.complex128)
                for chnl in self.data_loader.channels:
                    if chnl != self._tx_channel and chnl not in self._excluded_channels:
                        ipp += self.data_loader.read(chnl, start_sample, read_length)
            else:
                ipp = self.data_loader.read(self._rx_channel, start_sample, read_length)
            # Extract rx samples
            rx = ipp[self.pro_params.rx_stencil].copy()
        else:
            ipp = np.zeros((len(self.exp_params.rx_channels), read_length), dtype=np.complex128)
            for i, chnl in enumerate[int | str](self.data_loader.channels):
                ipp[i, :] = self.data_loader.read(chnl, start_sample, read_length)
            # Extract rx samples
            rx = ipp[:, self.pro_params.rx_stencil].copy()

        # Extracting tx data
        if self._tx_channel != self._rx_channel and self._tx_channel is not None:
            tx = self.data_loader.read(self._tx_channel, start_sample, read_length)
            tx = np.broadcast_to(tx.reshape((tx.size, 1)), (tx.size, sub_resolution))
        elif self._tx_channel == self._rx_channel and self._tx_channel is not None:
            tx = ipp.copy()
            tx = np.broadcast_to(tx.reshape((tx.size, 1)), (tx.size, sub_resolution))
        else:
            assert self.exp_params.code is not None, (
                "No code available from the metadata, not possible to simulate tx"
            )
            tx = tx_signal_model(
                code=self.exp_params.code,
                baud_length_usec=self.exp_params.baud_length_usec,
                t_samp_usec=self.exp_params.t_samp_usec,
                tx_start_samp=int(self.exp_params.t_tx_start_usec / self.exp_params.t_samp_usec),
                start_samp=(start_sample % self.exp_params.ipp_samps) - self.cfg_params.samp_offset,
                read_length=read_length,
                ipp_samps=self.exp_params.ipp_samps,
                sub_resolution=sub_resolution,
                kind="linear",
            )
        tx = tx[self.pro_params.tx_stencil, :]

        return ExtractedSignals(
            tx=tx.astype(np.complex64), rx=rx.astype(np.complex64), ipp=ipp.astype(np.complex64)
        )
