_COMM = None
_IMPORTED = False


class Comm:
    pass


def import_mpi():
    global _COMM, _IMPORTED
    if not _IMPORTED:
        try:
            from mpi4py import MPI
            _COMM = MPI.COMM_WORLD
        except ImportError:
            if _COMM is None:
                _COMM = Comm()
                _COMM.rank = 0
                _COMM.size = 1
        _IMPORTED = True


def get_mpi():
    global _COMM
    return _COMM
