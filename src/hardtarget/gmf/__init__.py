from .gmf_numpy import gmfnp, gmfnp_no_reduce

GMF_LIBS = {
    "numpy": gmfnp,
    "numpy_no_reduce": gmfnp_no_reduce,
}

# TODO: add fine optimization!!!!!!!!!!!!!!!!!!!!!!!

try:
    from .gmf_c import gmfc
except ImportError:
    gmfc = None
else:
    GMF_LIBS["c"] = gmfc

try:
    from .gmf_cuda import gmfcu
except ImportError:
    gmfcu = None
else:
    GMF_LIBS["cuda"] = gmfcu
