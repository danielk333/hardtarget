# # Visualization of raw data in both range and frequency:
# ---
# Simple example on how the raw rti and fti function can be utilized.

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from radardef import RadarDef

import hardtarget as ht

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

# ## Prerequisites
# ---

# Data to analyse

tmp_dir = tempfile.TemporaryDirectory()
raw_path = Path(tmp_dir.name) / "raw"
raw_path.mkdir(parents=True, exist_ok=True)
converted_path = Path(tmp_dir.name) / "converted"
raw_data = utils.download_test_data(raw_path)

# Convert the data to a usable format.

data = utils.convert_test_data(raw_data, converted_path)[0]

# Read data

reader = RadarDef().load_data(data, cache=False)
assert reader is not None
raw = reader.read()

# Plot for both range and frequency

fig, axes = plt.subplots(2, 1)
ht.plotting.rti(axes[0], reader)
axes[0].set_title("Range [rti]")
ht.plotting.fti(axes[1], reader)
axes[1].set_title("Frequency [fti]")
plt.show()
