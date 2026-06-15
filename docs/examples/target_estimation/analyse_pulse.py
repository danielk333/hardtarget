# # Analyse individual radar pulse
# ---
# How to extract and analyse an individual radar pulse

import argparse
from pathlib import Path
import tempfile

import numpy as np
from matplotlib import pyplot as plt

from hardtarget.target_estimation.gmf.types import GMFCfgParams
from hardtarget import plotting
from hardtarget.plotting.raw_data_plots import extract_requested_range_gates
from hardtarget.types import Bounds
from hardtarget.process.utils import sample_interval_to_closest_ipp
from radardef.types import BeamType, EiscatUHFLocation
from hardtarget.utils.time_conversion import time_interval_to_sample_bound, ts_from_str
import radardef

cfg = GMFCfgParams(
    n_ipp=1,
    ipp_offset=0,
    samp_offset=3,
    min_range_gate=4000,
    max_range_gate=8000,
    min_acceleration=0,
    max_acceleration=0,
    range_gate_step=1,
    range_gate_sub_resolution=10,
    frequency_decimation=1,
    num_cohints_per_file=2,
    node_gpus=1,
    acceleration_steps=1,
)
# ##Prerequisites
# ---

# Data to analyse

parser = argparse.ArgumentParser()
parser.add_argument("data_file", type=Path)
args = parser.parse_args()

# Temp or given dir to store precision orbit data
tmp_dir = tempfile.TemporaryDirectory()
output_dir = Path(tmp_dir.name) / "analysed"

# Time object is noticed in eiscat data
start_time_str = "2024-07-04T10:21:17.500"
end_time_str = "2024-07-04T10:21:20.000"

# Measurement source station
radar_station = radardef.EiscatUHF(location=EiscatUHFLocation.TROMSO, beam_type=BeamType.CASSEGRAIN)

reader = radar_station.load_data(args.data_file)
assert reader is not None

# ## Plot
# ---
# So what is seen here is the power for each rx sample for each ipp on the logarithmic scale, for a specific
# set of ipps

start_time = int(ts_from_str(start_time_str) * 1e6)
end_time = int(ts_from_str(end_time_str) * 1e6)

request_bounds = time_interval_to_sample_bound(
    time_bounds=Bounds(int(reader.epoch_bounds.ts_start_usec), int(reader.epoch_bounds.ts_end_usec)),
    start_time=start_time,
    end_time=end_time,
    sample_rate=reader.experiment.sample_rate,
    relative_time=False,
)

samp_bounds = sample_interval_to_closest_ipp(request_bounds, reader.experiment.ipp_samps)

# extract data within bounds
n_samp = samp_bounds.end - samp_bounds.start
data_vec = reader.read(
    channel=reader.experiment.rx_channels, start_sample=samp_bounds.start, vector_length=n_samp
)

if data_vec.ndim > 1:
    data_vec = np.sum(data_vec, axis=0)


# define experiment tx and rx intervals
def usec_to_sample(t_usec: int) -> int:
    return int(t_usec / reader.experiment.t_samp_usec)


t_rx_start_samp = usec_to_sample(reader.experiment.t_rx_start_usec)
t_rx_end_samp = usec_to_sample(reader.experiment.t_rx_end_usec)
t_tx_start_samp = usec_to_sample(reader.experiment.t_tx_start_usec)
t_tx_end_samp = usec_to_sample(reader.experiment.t_tx_end_usec)
t_cal_on_samp = (
    usec_to_sample(int(reader.experiment.t_cal_on_usec)) if reader.experiment.t_cal_on_usec is not None else 0
)
t_cal_off_samp = (
    usec_to_sample(int(reader.experiment.t_cal_off_usec))
    if reader.experiment.t_cal_off_usec is not None
    else 0
)

range_t = t_tx_start_samp / reader.experiment.sample_rate
samp_vec = np.arange(reader.experiment.ipp_samps)
rt_vec = np.arange(t_rx_end_samp - t_rx_start_samp) * reader.experiment.t_samp_usec - range_t

mat_shape = (data_vec.size // reader.experiment.ipp_samps, reader.experiment.ipp_samps)
data_ipp_vec = data_vec.reshape(mat_shape).T

il0_rg0, il0_rg1 = extract_requested_range_gates(
    cfg.min_range_gate, cfg.max_range_gate, "sample", reader.experiment
)
data_ipp_vec = data_ipp_vec[il0_rg0:il0_rg1, :]

fig, axes = plt.subplots(2, 1)
axes[0].plot(np.real(data_ipp_vec[:, 0]))
axes[0].plot(np.imag(data_ipp_vec[:, 0]))


fig, ax = plt.subplots()
ax, handles = plotting.rti(
    ax,
    reader,
    start_time=start_time_str,
    end_time=end_time_str,
    axis_units=True,
    log=True,
    keep_tx=True,
)

plt.show()
tmp_dir.cleanup()
