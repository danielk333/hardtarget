"""C Type definitions"""

from typing import Iterable, Literal, TypeAlias

import numpy as np
import numpy.ctypeslib as npct

NDPOINTER_FLAGS: TypeAlias = Iterable[
    Literal[
        "C_CONTIGUOUS",
        "CONTIGUOUS",
        "C",
        "F_CONTIGUOUS",
        "FORTRAN",
        "F",
        "ALIGNED",
        "A",
        "WRITEABLE",
        "W",
        "OWNDATA",
        "O",
        "WRITEBACKIFCOPY",
        "X",
    ]
]

FLAGS_W: NDPOINTER_FLAGS = ["ALIGNED", "C_CONTIGUOUS", "WRITEABLE"]
FLAGS_RO: NDPOINTER_FLAGS = ["ALIGNED", "C_CONTIGUOUS"]


def nptype(dtype: str, ndim: int, w: bool = False) -> type[npct._ndptr]:
    """Convenience function for generating appropriate C-type declarations for loaded shared libraries.

    See below links for more information:
     - https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
     - https://numpy.org/doc/stable/reference/routines.ctypeslib.html
    """
    flags = FLAGS_W if w else FLAGS_RO
    return npct.ndpointer(np.dtype(dtype), ndim=ndim, flags=flags)
