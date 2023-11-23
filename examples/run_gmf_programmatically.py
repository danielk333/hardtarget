from pathlib import Path
from pprint import pprint
import hardtarget

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

base_path = Path("/home/danielk/data/spade/")
drf_path = base_path / "beamparks_raw" / "leo_bpark_2.1u_NO@uhf_drf"
rx_channel = "uhf"
config_path = Path("./examples/cfg/test.ini").resolve()
output_path = base_path / "beamparks_analyzed" / "leo_bpark_2.1u_NO@uhf_c"

# process
results = hardtarget.compute_gmf(
    rx=(drf_path, rx_channel),
    tx=(drf_path, rx_channel),
    config=config_path,
    job={"idx": rank, "N": size},
    gmflib="c",
    clobber=True,
    output=output_path,
    start_time="2021-04-12T12:15:40",
    end_time="2021-04-12T12:16:10",
    relative_time=False,
    progress=True,
    progress_position=rank,
)

pprint(results)
