import unittest
import configparser
from hardtarget.config import unpack_config
from hardtarget.analysis import analyze_params


TEST_GMF_CONFIG = """
[radar-experiment]
sample_rate=100000
ipp=2000
tx_pulse_length=202
doppler_sign=-1.0
radar_frequency=500e6
round_trip_range=false
rx_channel='32m'
tx_channel='32m'

[signal-processing]
n_ipp=5
frequency_decimation=2
n_range_gates=1800
range_gate_0=100
range_gate_step=1
ground_clutter_length=50
min_acceleration=0.0
max_acceleration=250.0
acceleration_resolution=0.2
snr_thresh=10.0
num_cohints_per_file=100
start_time=1515067356
end_time=1515067956
gmflib='c'

[bogus]
test=False
"""

CONFIG_KEYS = [
    "sample_rate",
    "ipp",
    "tx_pulse_length",
    "doppler_sign",
    "radar_frequency",
    "round_trip_range",
    "rx_channel",
    "tx_channel",
    "n_ipp",
    "frequency_decimation",
    "n_range_gates",
    "range_gate_0",
    "range_gate_step",
    "ground_clutter_length",
    "min_acceleration",
    "max_acceleration",
    "acceleration_resolution",
    "snr_thresh",
    "num_cohints_per_file",
    "start_time",
    "end_time",
    "gmflib",
]


class test_Config(unittest.TestCase):
    def test_unpack_config(self):
        config = configparser.ConfigParser()
        config.read_string(TEST_GMF_CONFIG)

        assert "bogus" in config
        assert config.getboolean("bogus", "test") is False

        params = unpack_config(config)

        for key in CONFIG_KEYS:
            assert key in params
        assert "test" not in params

        # a few samples
        assert isinstance(params["n_ipp"], int)
        assert isinstance(params["gmflib"], str)
        assert isinstance(params["radar_frequency"], float)

    def test_check_params(self):
        config = configparser.ConfigParser()
        config.read_string(TEST_GMF_CONFIG)
        params = unpack_config(config)
        assert analyze_params.check_params(params)


if __name__ == "__main__":
    unittest.main()
