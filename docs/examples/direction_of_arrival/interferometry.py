# # Interferometry
# ---
# How to trigger analysis and then inspect the output

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from radardef import ExpParams, Mu, RadarDef

from hardtarget import direction_of_arrival, load_analysed_data, plotting
from hardtarget.interferometry.types import DOACfgParams, DOAProParams, DOAVars

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "test.ini"

# ##Prerequisites
# ---

# Data to analyse

tmp_dir = tempfile.TemporaryDirectory()
raw_path = Path(tmp_dir.name) / "raw"
raw_path.mkdir(parents=True, exist_ok=True)
converted_path = Path(tmp_dir.name) / "converted"
raw_data = utils.download_test_data(Path(tmp_dir.name) / "raw")

# Convert the data to a usable format.

radars = RadarDef()
source_format = radars.get_source_format(raw_data)
target_formats = radars.available_target_formats(source_format)
converted_files = RadarDef().convert(raw_data, target_formats[0], str(converted_path))
data = converted_files[0]

reader = RadarDef().load_data(data)
assert reader is not None
mu_station = Mu()

# ## Analyse data
# ---
# Minimal configuration for analysis, progress, start_time, end_time and relative time are optionals.
# In this case we choose to visualize the 500 first ipps of the measurement

output_path = Path(tmp_dir.name) / "analysed"
start_time = int(reader.meta.experiment.t_ipp_usec * 40)
end_time = start_time + int(reader.meta.experiment.t_ipp_usec * 150)
# Alternative object
# start_time = int(reader.meta.experiment.t_ipp_usec * 3450)  # 3350
# end_time = start_time + int(reader.meta.experiment.t_ipp_usec * 150)
cfg = DOACfgParams(
    n_ipp=1,
    ipp_offset=0,
    min_range_gate=81,
    max_range_gate=138,
    range_gate_step=1,
    num_cohints_per_file=500,
    elevation_limit=0,
    resolution=100,
)

# Run analysis

direction_of_arrival(
    path=data,
    config=cfg,
    array_beam=mu_station.beam,
    beam_params=mu_station.beam_parameters,
    output=output_path,
    start_time=start_time,
    end_time=end_time,
    relative_time=True,
    progress=True,
)

# Load analysed data

data_generator = load_analysed_data(output_path)
output: tuple[DOAVars, ExpParams, DOACfgParams, DOAProParams] = list(data_generator)[0]
out_data, exp_params, cfg_params, pro_params = output

# Plot results

fig, ax = plt.subplots(3, 2)
ax, ani = plotting.plot_direction_of_arrival(fig, ax, out_data, pro_params, 10.0)
#
ax[2, 1].set_title("Raw data (summed channels)")
_, handles = plotting.rti(
    ax[2, 1],
    reader,
    start_time=start_time,
    end_time=end_time,
    axis_units=True,
    log=True,
    relative_time=True,
    colorbar=False,
)
#
plt.show()
