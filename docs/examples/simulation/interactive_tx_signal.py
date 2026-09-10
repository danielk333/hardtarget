# # Interactive tx simulation
# ---

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from radardef.radar_stations.eiscat.experiments import load_radar_code

from hardtarget.constants import ReceiverChainModel
from hardtarget.data_simulation.tx_model import tx_signal_model

# First we define a simple code
# ```
#   ‾‾‾‾‾|  |‾‾| |‾| |‾
#        |__|  |_| |_|
# ```
# TODO: something is wrong with the mu filter
barker13 = np.array(
    [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
    dtype=np.float64,
)

# fir_filter = ReceiverChainModel.mu2004
fir_filter = ReceiverChainModel.b414d15_gaus
signal_decimation = {
    ReceiverChainModel.b414d15_gaus: 15,
    ReceiverChainModel.mu2004: 120,
    ReceiverChainModel.none: 1,
}
sample_time_usec = {
    ReceiverChainModel.b414d15_gaus: 1,
    ReceiverChainModel.mu2004: 6,
    ReceiverChainModel.none: 6,
}
baud_lengths_usec = {
    ReceiverChainModel.b414d15_gaus: 30,
    ReceiverChainModel.mu2004: 12,
    ReceiverChainModel.none: 12,
}
codes = {
    ReceiverChainModel.b414d15_gaus: load_radar_code("leo_bpark")[0, :],
    ReceiverChainModel.mu2004: barker13,
    ReceiverChainModel.none: barker13,
}

tx_samples = len(codes[fir_filter]) * int(baud_lengths_usec[fir_filter] / sample_time_usec[fir_filter])
ipp_samps = tx_samples * 5

model_len = np.linspace(0, 1, tx_samples * 2)
model_len_orig = np.linspace(0, 1, tx_samples * signal_decimation[fir_filter] * 2)
sub_resolution = np.arange(0, 2, 0.01)

tx_base = tx_signal_model(
    code=codes[fir_filter],
    baud_length_usec=baud_lengths_usec[fir_filter],
    t_samp_usec=sample_time_usec[fir_filter] / signal_decimation[fir_filter],
    start_samp=tx_samples * signal_decimation[fir_filter] * 0.5,
    ipp_samps=ipp_samps * signal_decimation[fir_filter],
    read_length=tx_samples * signal_decimation[fir_filter] * 2,
    filt=ReceiverChainModel.none,
    sub_resolution=sub_resolution * signal_decimation[fir_filter],
    bandwidth=1e6,
)
filter_options = (ReceiverChainModel.b414d15_gaus, ReceiverChainModel.mu2004, ReceiverChainModel.none)
signals = {}


def signal_for(selected_filter):
    if selected_filter not in signals:
        signals[selected_filter] = tx_signal_model(
            code=codes[fir_filter],
            baud_length_usec=baud_lengths_usec[fir_filter],
            t_samp_usec=sample_time_usec[fir_filter],
            start_samp=tx_samples * 0.5,
            ipp_samps=ipp_samps,
            read_length=tx_samples * 2,
            filt=selected_filter,
            sub_resolution=sub_resolution,
            bandwidth=1e6,
        )
    return signals[selected_filter]


tx = signal_for(fir_filter)
ind = 0
zoomed = True
fig, ax = plt.subplots()

# Center the view on a phase change and show one baud width.  Using sample
# coordinates makes the extent independent of the length of the radar code.
transition_baud = np.flatnonzero(np.diff(codes[fir_filter]) != 0)[0] + 1
transition_sample = (
    tx_samples * 0.5 + transition_baud * baud_lengths_usec[fir_filter] / sample_time_usec[fir_filter]
)
zoom_start = transition_sample - 0.5 * baud_lengths_usec[fir_filter] / sample_time_usec[fir_filter]
zoom_stop = transition_sample + 0.5 * baud_lengths_usec[fir_filter] / sample_time_usec[fir_filter]
sample = model_len * tx_samples * 2
sample_orig = model_len_orig * tx_samples * 2
extent = np.logical_and(sample >= zoom_start, sample <= zoom_stop)
extent_o = np.logical_and(sample_orig >= zoom_start, sample_orig <= zoom_stop)

(ls,) = ax.plot(sample[extent], np.real(tx[extent, ind]), "o-", label="filtered")
(ls_base,) = ax.plot(sample_orig[extent_o], np.real(tx_base[extent_o, ind]), "-", label="original")
ax.set_xlim(zoom_start, zoom_stop)
ax.set_xlabel("Sample")
ax.legend()


def update_title():
    sample_rate_mhz = 1 / sample_time_usec[fir_filter]
    ax.set_title(
        f"Decimation: {signal_decimation[fir_filter]} | "
        f"Sample rate: {sample_rate_mhz:g} MHz | "
        f"Baud length: {baud_lengths_usec[fir_filter]:g} us"
    )


def draw():
    if zoomed:
        ls.set_data(sample[extent], np.real(tx[extent, ind]))
        ls_base.set_data(sample_orig[extent_o], np.real(tx_base[extent_o, ind]))
        ax.set_xlim(zoom_start, zoom_stop)
    else:
        ls.set_data(sample, np.real(tx[:, ind]))
        ls_base.set_data(sample_orig, np.real(tx_base[:, ind]))
        ax.set_xlim(sample[0], sample[-1])
    ax.relim()
    ax.autoscale_view(scalex=False)
    update_title()
    fig.canvas.draw_idle()


def update_d(val):
    global ind
    ind = np.argmin(np.abs(sub_resolution - val))
    draw()


fig.tight_layout()
fig.subplots_adjust(bottom=0.14, top=0.86)
update_title()

axcolor = "lightgoldenrodyellow"
ax_d = plt.axes([0.1, 0.05, 0.2, 0.03], facecolor=axcolor)
s_d = Slider(ax_d, "Offset", 0, 2, valinit=0, valstep=0.01)
s_d.on_changed(update_d)

# I could not fina a dropdown widget, so this is a button to reveal a small
# stack of option buttons and hide them again after a selection is made.
menu_axes = [plt.axes([0.72, 0.80 - i * 0.045, 0.22, 0.04]) for i in range(len(filter_options))]
menu_buttons = [Button(menu_ax, option.value) for menu_ax, option in zip(menu_axes, filter_options)]
for menu_ax in menu_axes:
    menu_ax.set_visible(False)

ax_filter = plt.axes([0.72, 0.90, 0.22, 0.045])
filter_button = Button(ax_filter, f"Filter: {fir_filter.value}")

ax_zoom = plt.axes([0.48, 0.90, 0.20, 0.045])
zoom_button = Button(ax_zoom, "Show full signal")


def toggle_filter_menu(_event):
    visible = not menu_axes[0].get_visible()
    for menu_ax in menu_axes:
        menu_ax.set_visible(visible)
    fig.canvas.draw_idle()


def select_filter(selected_filter):
    def update_filter(_event):
        global fir_filter, tx
        fir_filter = selected_filter
        tx = signal_for(fir_filter)
        filter_button.label.set_text(f"Filter: {fir_filter.value}")
        for menu_ax in menu_axes:
            menu_ax.set_visible(False)
        draw()

    return update_filter


def toggle_zoom(_event):
    global zoomed
    zoomed = not zoomed
    zoom_button.label.set_text("Show full signal" if zoomed else "Zoom to transition")
    draw()


filter_button.on_clicked(toggle_filter_menu)
zoom_button.on_clicked(toggle_zoom)
for menu_button, option in zip(menu_buttons, filter_options):
    menu_button.on_clicked(select_filter(option))

plt.show()
