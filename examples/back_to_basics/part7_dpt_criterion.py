"""

# TODO: finish writing description here

Criterion for discrete polynomial-phase transformation determination

$$
|a_M| \leq \frac{\pi f_s^{M}}{M! \tau^{M-1}}
$$

we have 3 parameters we can vary, lets for simplicity fix $f_s$ and fix $\tau$ to

$$
\tau = \frac{N}{2}
$$

this gives

$$
|a_M| \leq \frac{ 2^{M-1} \pi f_s^{M}}{M! N^{M-1}}
$$

"""
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt


def a_abs(M, N, f_s):
    # |a_M| < limit, hence limit is at a_M = limit
    return np.pi * 2**(M - 1) * f_s**M / (special.factorial(M) * N**(M - 1))


f_s = int(1e6)
ipp = int(2e4)
M = 2
N = np.arange(ipp, ipp*20, ipp)
fig, ax = plt.subplots()

ax.semilogy(N, a_abs(M, N, f_s))

ax.axhline(10.0, c="r")

plt.show()
