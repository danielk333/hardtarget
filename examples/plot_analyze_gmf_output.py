"""
Plot GMF results
=================

hardtarget convert eiscat ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf \
    -o ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf

"""

from pathlib import Path
import configparser
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py

# Give the data path as the first argument to this example
parser = argparse.ArgumentParser(description="Plot the analyze gmd output")
parser.add_argument("data_folder")
args = parser.parse_args()

h5_files = list(Path(args.data_folder).glob("**/*.h5"))
h5_files.sort()
print(f"{len(h5_files)} files found")

inds = list(range(len(h5_files)))

print("Loading data...")
t_vecs = None
for ind in inds:
    print(h5_files[ind])

    with h5py.File(h5_files[ind], "r") as hf:
        gmf = hf["gmf"][()]

        coh_ints = np.arange(gmf.shape[0])

        r_inds = np.argmax(gmf, axis=1)
        v_inds = hf["range_rate_index"][()]
        a_inds = hf["acceleration_index"][()]

        r = hf["ranges"][()]
        v = hf["range_rates"][()]
        a = hf["accelerations"][()]

    n_cohints = gmf.shape[0]
    if ind == inds[0]:
        t_vec = np.arange(n_cohints)
    else:
        t_vec = np.arange(t_vecs[-1], t_vecs[-1] + n_cohints)
    r_vec = r[r_inds]
    v_vec = v[v_inds[coh_ints, r_inds]]
    a_vec = a[a_inds[coh_ints, r_inds]]
    g_vec = gmf[coh_ints, r_inds]

    if ind == inds[0]:
        gmf_vec = gmf
        t_vecs = t_vec
        r_vecs = r_vec
        v_vecs = v_vec
        a_vecs = a_vec
        g_vecs = g_vec
    else:
        gmf_vec = np.append(gmf_vec, gmf, axis=0)
        t_vecs = np.append(t_vecs, t_vec)
        r_vecs = np.append(r_vecs, r_vec)
        v_vecs = np.append(v_vecs, v_vec)
        a_vecs = np.append(a_vecs, a_vec)
        g_vecs = np.append(g_vecs, g_vec)

fig, ax = plt.subplots()
ax.pcolormesh(gmf_vec.T)
plt.show()


fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(t_vecs, r_vecs)
axes[0, 1].plot(t_vecs, v_vecs)
axes[1, 0].plot(t_vecs, a_vecs)
axes[1, 1].plot(t_vecs, g_vecs)

plt.show()
