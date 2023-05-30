import pytest
import numpy

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    MPI = None
    rank = 0
    size = 1

@pytest.mark.skip
def test_mpi():
    print(f"Hello, World! I am process {rank} of {size}.")

if MPI is not None:
    MPI.Finalize()