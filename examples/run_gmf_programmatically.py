#!/usr/bin/env python
"""
Run analysis programmatically
==============================

This script shows how to call the `compute_gmf` function directly.

For developers this is also particularly useful since it can be used to
debug the backends developed in different programming languages.

For example, to debug the C-implementation using GNU debugger simply

```
gdb --args python ./examples/run_gmf_programmatically.py
```

to start gdb targeting the python binary which is set to run this example.
Then we need to load the debug symbols from the correct binary to set
breakpoints, we can do this (if this is installed in developer-mode using the
pip `-e` option) by typing these commands into gdb

```
add-symbol-file ./src/hardtarget/gmf/gmfclib.cpython-311-x86_64-linux-gnu.so
break gmf.c:gmf
run
```

which will run this file and halt execution at the `gmf` function in the C
implementation in `gmf.c` and allow you to debug the compiled c-code as it is
exactly loaded and run by python.

"""


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
