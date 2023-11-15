import pathlib
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as consts


def get_noise_floor(path):
    fl = list(path.glob("*/gmf*.h5"))
    fl.sort()
    print(fl)
    floors = []
    for f in fl:
        print(f)
        h = h5py.File(f, "r")
        gmf = h["gmf"][()]
        floors.append(np.nanmedian(gmf, axis=0))
        h.close()
    floors = np.array(floors)
    return np.nanmedian(floors, axis=0)


def plot_gmfs(path, nf):
    fl = list(path.glob("*/gmf*.h5"))
    fl.sort()
    print(len(fl))
    for f in fl:
        print(f)
        h = h5py.File(f, "r")
        gmf = h["gmf"][()]
        v = h["v"][()]
        a = h["a"][()]
        breakpoint()
        for i in range(gmf.shape[0]):
            gmf[i, :] = (gmf[i, :] - nf) / nf

        if np.max(gmf) > 10.0:
            plt.subplot(221)
            plt.plot(np.max(gmf, axis=1))
            plt.subplot(222)
            dr = consts.c / h.attrs["sample_rate"] / 2.0 / 1e3
            plt.plot(dr * h["vector_params"]["rgs"][np.argmax(gmf, axis=1)], ".")
            plt.subplot(223)

            vels = []
            for i in range(gmf.shape[0]):
                mi = np.argmax(gmf[i, :])
                vels.append(v[i, mi])

            plt.plot(vels, ".")
            plt.subplot(224)
            ass = []
            for i in range(gmf.shape[0]):
                mi = np.argmax(gmf[i, :])
                ass.append(a[i, mi])

            plt.plot(ass, ".")
            plt.show()
        h.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(sys.argv[1])
        path = pathlib.Path(sys.argv[1]).resolve()
    else:
        print("Provide path as command line option")
        exit(0)
    nf = get_noise_floor(path)
    nf[nf < 1] = 1.0
    plt.plot(nf)
    plt.show()
    plot_gmfs(path, nf)
