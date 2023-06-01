from .gmf_numpy import gmf_numpy

GMF_LIBS = {
    'numpy': gmf_numpy,
}

try:
    from .gmf_c import gmf_c
except ImportError:
    gmf_c = None
else:
    GMF_LIBS['c'] = gmf_c

try:
    from .gmf_cuda import gmf_cuda
except ImportError:
    gmf_cuda = None
else:
    GMF_LIBS['cuda'] = gmf_cuda