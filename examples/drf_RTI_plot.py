import matplotlib.pyplot as plt
import digital_rf as drf
import hardtarget
import numpy as np

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf/"
drf_reader = drf.DigitalRFReader(target)
bounds = drf_reader.get_bounds("drf")
data_vec = drf_reader.read_vector_1d(bounds[0] + 400000, 600000, "drf")

fig, ax = plt.subplots()
ax.plot(np.abs(data_vec))

fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    drf_reader,
    "drf",
    # start_time="2021-04-12T12:03:55",
    # end_time="2021-04-12T12:04:02",
)

plt.show()
