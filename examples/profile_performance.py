"""
Profiling
==========
"""

import hardtarget
from pathlib import Path

base_path = Path("/home/danielk/data/spade/")
drf_path = base_path / "beamparks_raw" / "leo_bpark_2.1u_NO@uhf_drf"
rx_channel = "uhf"
config_path = Path("./examples/cfg/test.ini").resolve()

# Does not work if yappi is not installed
hardtarget.profile()

libs = ["numpy_daf", "cuda", "c", "numpy"]
# Do computation
for lib in libs:
    # process
    results = hardtarget.compute_gmf(
        rx=(drf_path, rx_channel),
        tx=(drf_path, rx_channel),
        config=config_path,
        job={"idx": 0, "N": 1},
        gmflib=lib,
        clobber=True,
        output=None,  # Then we get data back in "results instead"
        start_time=0,
        end_time=0.2,  # Just one coherent integration
        relative_time=True,
        progress=True,
    )

    # print and clear
    stats, total = hardtarget.get_profile()
    print(f"LIB={lib}: total time = {total:.4f} [s]")
    hardtarget.print_profile(stats, total=total, max_rows=5)
    hardtarget.profile_clear()
