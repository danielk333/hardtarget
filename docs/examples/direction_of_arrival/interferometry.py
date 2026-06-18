# # Direction of arrival
# ---
# How to trigger analysis and then inspect the output

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from radardef import ExpDef, Mu, RadarDef

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

data = utils.convert_test_data(raw_data, converted_path)[0]

reader = RadarDef().load_data(data, cache=False)
assert reader is not None
mu_station = Mu()

# ## Analyse data
# ---
# Minimal configuration for analysis, progress, start_time, end_time and relative time are optionals.
# In this case we choose to visualize the 500 first ipps of the measurement

output_path = Path(tmp_dir.name) / "analysed"
start_time = int(reader.exp_def.t_ipp_usec * 50)
end_time = start_time + int(reader.exp_def.t_ipp_usec * 120)
cfg = DOACfgParams(
    n_ipp=1,
    ipp_offset=0,
    min_range_gate=81,
    max_range_gate=138,
    range_gate_step=1,
    num_cohints_per_file=500,
    elevation_limit=0,
    resolution=150,
    cache=False,
)

# Run analysis
direction_of_arrival(
    data=data,
    config=cfg,
    array_beam=mu_station.beam,
    beam_params=mu_station.beam_parameters,
    output=output_path,
    start_time=start_time,
    end_time=end_time,
    relative_time=True,
)

# Load analysed data

data_generator = load_analysed_data(output_path)
output: tuple[DOAVars, ExpDef, DOACfgParams, DOAProParams] = list(data_generator)[0]
out_data, exp_def, cfg_params, pro_params = output

# ## Plot results
# ---
# Plot the estimated path over the beam and the azimuth and elevation.

fig, ax = plt.subplots(2, 2)
ax = plotting.plot_direction_of_arrival(ax, out_data, pro_params, 10.0)
fig.set_size_inches(10, 10)
#
plt.show()
