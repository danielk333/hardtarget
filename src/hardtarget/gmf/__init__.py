from .gmf_numpy import gmfnp

GMF_LIBS = {
    "numpy": gmfnp,
}

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
