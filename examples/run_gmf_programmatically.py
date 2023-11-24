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

If one is debugging the cuda code, things get a bit harder but it is still
possible using `cuda-gdb` instead of `gdb`, then simply

```
add-symbol-file ./src/hardtarget/gmf/gmfcudalib.cpython-311-x86_64-linux-gnu.so
break gmfgpu.cu:gmf
run
```

and it becomes possible to step into cuda kernel functions. In this case there
are some special commands avalible like

```
info cuda threads
info cuda kernels
info cuda threads kernel 0
```

to access variables one might have to use the correct memory location specifiers
such as @global, @shared, @local, @generic, @texture, and @parameter. However,
usually it is sufficient to use the regular syntax like `print *rgs@10`

WARNING: Sometimes its possible to mess up the GPU temporarily during debugging
leaving halted threads and memory leaks if not careful. In this case, reboot to
return to original performance.

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

# gmflib = "c"
gmflib = "cuda"

base_path = Path("/home/danielk/data/spade/")
drf_path = base_path / "beamparks_raw" / "leo_bpark_2.1u_NO@uhf_drf"
rx_channel = "uhf"
config_path = Path("./examples/cfg/test.ini").resolve()
output_path = base_path / "beamparks_analyzed" / f"leo_bpark_2.1u_NO@uhf_{gmflib}"

# process
results = hardtarget.compute_gmf(
    rx=(drf_path, rx_channel),
    tx=(drf_path, rx_channel),
    config=config_path,
    job={"idx": rank, "N": size},
    gmflib=gmflib,
    clobber=True,
    output=output_path,
    start_time="2021-04-12T12:15:40",
    end_time="2021-04-12T12:16:10",
    relative_time=False,
    progress=True,
    progress_position=rank,
)

pprint(results)
