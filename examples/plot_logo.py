"""

MAKE DOC HERE
=============

todo

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

plt.style.use('dark_background')

signal_len = 20000
sample_rate = int(1e4)
box_size = 6000
cycle_size = 10000
boxes = signal_len // cycle_size
frequency = 1e1
sig = 0.3

signal = np.zeros((signal_len, ), dtype=np.complex128)
stensil = np.full((signal_len, ), True, dtype=bool)
t = np.arange(signal_len) / sample_rate

for ind in range(boxes):
    stensil[(cycle_size*ind):(cycle_size*ind + box_size)] = False

xir = np.random.randn(signal_len)
xii = np.random.randn(signal_len)
signal = np.exp(2j * np.pi * frequency * t)
signal[stensil] = 0
signal0 = signal.copy()/np.max(np.abs(signal))
signal += sig*(xir + 1j*xii)
signal = signal/np.max(np.abs(signal))

fvec = fft.fftshift(fft.fftfreq(signal_len, d=1.0/sample_rate))
spec = np.abs(fft.fftshift(fft.fft(signal)))

df = 5e1

fig, ax = plt.subplots(figsize=(6, 6))

posv = np.arange(signal_len)
dec = 80
ax.plot(posv[::dec], np.real(signal[::dec][::-1])*0.5 + 2, c="#a3e5f4")
ax.plot(posv, np.real(signal0)*0.5 + 1, c="#3fea19")

sub_spec = spec[np.logical_and(fvec > frequency - df, fvec < frequency + df)]
sub_spec = sub_spec/np.max(np.abs(sub_spec))
ax.plot(np.linspace(0, signal_len, num=len(sub_spec)), sub_spec*2, c="#41c4ec", lw=3)

# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()

# fig.savefig('logo.svg', bbox_inches='tight')
# fig.savefig('logo.png', bbox_inches='tight')
# plt.gcf().set_size_inches(2, 2)
# fig.savefig('favicon.png', bbox_inches='tight')

plt.show()
