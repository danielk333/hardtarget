from typing import NamedTuple

SIMULATION_PARAMS = [
    "epoch",
    "start_time",
    "end_time",
    "noise_sigma",
]

SIMULATION_DATA = [
    "ranges",
    "velocities",
    "accelerations",
    "snrs",
    "times",
]


class DRFSimParams(NamedTuple):
    epoch: str | int = "2021-04-12T12:15:40"
    start_time_us: int = 0
    end_time_us: int = 10000
    target_start_time_us: int = 0
    target_end_time_us: int = 10000
    noise_sigma: float = 0
    tx_amp: float = 1
