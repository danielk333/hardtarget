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


def load_simulation_params(file):
    """Load simulation parameters from a h5 file
    """
    raise NotImplementedError()


def save_simulation_params(file):
    raise NotImplementedError()
