"""
Plot GMF results
=================

hardtarget convert eiscat ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf \
    -o ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf

"""

from pathlib import Path
import configparser
import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py

# Give the data path as the first argument to this example
parser = argparse.ArgumentParser(description="Plot the analyze gmd output")
parser.add_argument("data_folder")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read("./examples/cfg/test.ini")

n_ranges = config.getint("signal-processing", "n_range_gates")
n_cohints = config.getint("signal-processing", "num_cohints_per_file")
sample_rate = config.getint("radar-experiment", "sample_rate")

h5_files = list(Path(args.data_folder).glob("**/*.h5"))
h5_files.sort()
print(f"{len(h5_files)} files found")

# The target event
# dt = datetime.datetime(2021, 4, 12, 12, 1, 33, 600000)
dt = datetime.datetime(2021, 4, 12, 12, 15, 57, 400000)
time_stamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
print(f"Event should be located at {time_stamp}...")
sample_stamp = time_stamp * sample_rate

samples_times = [int(Path(x).name.split("-")[1].split(".")[0]) for x in h5_files]
samples_times = np.array(samples_times)
dt = samples_times - sample_stamp
file_ind = np.argmin(np.abs(dt))

print(f"Found files {dt[file_ind]/sample_rate} sec off target event")

inds = list(range(file_ind - 5, file_ind + 15))

print("Loading data...")
t_vecs = None
for ind in inds:
    print(h5_files[ind])

    with h5py.File(h5_files[ind], "r") as hf:
        gmf = hf["gmf"][()]
        r = hf["r"][()]
        v = hf["v"][()]
        a = hf["a"][()]
    r_inds = np.argmax(gmf, axis=1)

    if ind == inds[0]:
        t_vec = np.arange(n_cohints)
    else:
        t_vec = np.arange(t_vecs[-1], t_vecs[-1] + n_cohints)
    r_vec = r[r_inds]
    v_vec = np.array([v[ind, r_inds[ind]] for ind in range(n_cohints)])
    a_vec = np.array([a[ind, r_inds[ind]] for ind in range(n_cohints)])
    g_vec = np.array([gmf[ind, r_inds[ind]] for ind in range(n_cohints)])

    if ind == inds[0]:
        t_vecs = t_vec
        r_vecs = r_vec
        v_vecs = v_vec
        a_vecs = a_vec
        g_vecs = g_vec
    else:
        t_vecs = np.append(t_vecs, t_vec)
        r_vecs = np.append(r_vecs, r_vec)
        v_vecs = np.append(v_vecs, v_vec)
        a_vecs = np.append(a_vecs, a_vec)
        g_vecs = np.append(g_vecs, g_vec)


fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(t_vecs, r_vecs)
axes[0, 1].plot(t_vecs, v_vecs)
axes[1, 0].plot(t_vecs, a_vecs)
axes[1, 1].plot(t_vecs, g_vecs)

plt.show()
