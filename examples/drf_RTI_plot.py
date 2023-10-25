import matplotlib.pyplot as plt
import digital_rf as drf
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf/"
drf_reader = drf.DigitalRFReader(target)

fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    drf_reader,
    "uhf",
    start_time=2,
    end_time=8,
    relative_time=True,
)

plt.show()
