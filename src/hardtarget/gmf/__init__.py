from .gmf_numpy import gmfnp, gmfnp_optimize

GMF_GRID_LIBS = {
    "numpy": gmfnp,
}

GMF_OPTIMIZE_LIBS = {
    "numpy": gmfnp_optimize,
}

try:
    from .gmf_c import gmfc
except ImportError:
    gmfc = None
else:
    GMF_GRID_LIBS["c"] = gmfc
    # GMF_OPTIMIZE_LIBS["c"] = gmfc_optimize

try:
    from .gmf_cuda import gmfcu
except ImportError:
    gmfcu = None
else:
    GMF_GRID_LIBS["cuda"] = gmfcu
    # GMF_OPTIMIZE_LIBS["cuda"] = gmfcu_optimize