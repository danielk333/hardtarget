import numpy.ctypeslib as npct
import numpy as np

FLAGS_W = 'aligned, c_contiguous, writeable'
FLAGS_RO = 'aligned, c_contiguous'


def nptype(dtype, ndim, w=False):
    """Convenience function for generating appropriate C-type declarations for loaded shared libraries.

    See below links for more information:
     - https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
     - https://numpy.org/doc/stable/reference/routines.ctypeslib.html
    """
    flags = FLAGS_W if w else FLAGS_RO
    return npct.ndpointer(np.dtype(dtype), ndim=ndim, flags=flags)
