# Simulation of Array beam
# ---
# Simulate an array beam containing 25 channels where a object moves over the beam

import datetime as dt
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from radardef import Mu, RadarDef
from radardef.types import ExpDef

from hardtarget.data_simulation import simulate_h5
from hardtarget.plotting import rti

temp_dir = tempfile.TemporaryDirectory()


# ## Defining the experiment definition
# ---

exp_params = ExpDef(
    name="Mu",
    radar_frequency=46.5,
    t_ipp_usec=3120,
    t_samp_usec=6,
    t_rx_start_usec=486,
    t_rx_end_usec=486 + 6 * 85,
    t_tx_start_usec=0,
    t_tx_end_usec=13 * 2 * 6,
    baud_length_usec=12,
    code=np.array(
        [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
        dtype=np.float64,
    ),
    rx_channels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    samples_per_file=10000000,
)

# ## Define a objects trajectory function,
# ---


def trajectory_func(t: npt.NDArray) -> npt.NDArray:
    # Initial values
    r0: float = 220e3
    v0: float = -0.4e3
    a0: float = -0.20e3
    # Initial position over radar
    k0 = np.array([0.1, 0.1, 0.8])
    # 3d position
    x_start = k0 * r0
    # Velocity vector, object moving along x-axis
    v_vec = np.array([0.75, 0.25, 0])
    distance_traveled = v0 * t + a0 * 0.5 * t**2
    # trajectory
    return x_start[:, None] + v_vec[:, None] * distance_traveled[None, :]


# ## Simulate the experiment with the MU radars beam.
# ---

output_path = Path(temp_dir.name) / "sim"
station = Mu()
target_start_time_us = 350000
target_end_time_us = 600000
simulate_h5(
    output_dir=output_path,
    exp_params=exp_params,
    start_time=dt.datetime.now(),
    end_time=dt.datetime.now() + dt.timedelta(seconds=1),
    target_start_time=target_start_time_us,
    target_end_time=target_end_time_us,
    target_relative_time=True,
    trajectory_function=trajectory_func,
    noise_sigma=0,
    beam=station.beam,
    beam_params=station.beam_parameters,
)

# ## Visualise the Raw data
# ---

data_loader = RadarDef().load_data(output_path, experiment=exp_params)
assert data_loader is not None, "Failed to load data"
fig, ax = plt.subplots(2)
data = np.zeros(data_loader.read(1).shape, dtype=np.complex128)
for c in exp_params.rx_channels:
    data += data_loader.read(c)
ax[0].plot(data)
ax[1], handles = rti(
    ax[1],
    data_loader=data_loader,
)
plt.show()
temp_dir.cleanup()
