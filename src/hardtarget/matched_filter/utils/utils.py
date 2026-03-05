"""Matched filter utilities"""

import numpy as np
import numpy.typing as npt


def default_mf_vars_items(
    vector_size: tuple,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    vals = np.zeros(vector_size, dtype=np.float32)
    dc = np.zeros(vector_size, dtype=np.float32)
    v_ind = np.full(vector_size, -1, dtype=np.int32)
    a_ind = np.full(vector_size, -1, dtype=np.int32)

    return vals, dc, v_ind, a_ind
