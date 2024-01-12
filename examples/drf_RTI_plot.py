import matplotlib.pyplot as plt
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
reader, params = hardtarget.load_hardtarget_drf(target)


fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    reader,
    params,
    start_time="2021-04-12T12:15:53",
    end_time="2021-04-12T12:16:03",
    axis_units=True,
    clutter_removal=2500.0e-6,  # s
    log=True,
)

plt.show()
