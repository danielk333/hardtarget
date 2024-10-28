#!/usr/bin/env python
"""
Run analysis programmatically
==============================

This script shows how to call the `compute_gmf` function directly.

For developers this is also particularly useful since it can be used to
debug the backends developed in different programming languages.

For example, if i had some data located at

`/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf`

I can debug the C-implementation using GNU debugger simply by running

```
gdb --args python ./examples/run_gmf_programmatically__no_agl.py \
    /home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf \
    /home/danielk/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf_fgmf_c_debug \
    uhf --lib c
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

import argparse
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

HERE = Path(__file__).parent.absolute() / "cfg" / "test.ini"

parser = argparse.ArgumentParser()
parser.add_argument("drf_path")
parser.add_argument("output_path")
parser.add_argument("rx_channel")
parser.add_argument("--config", type=Path, default=HERE)
parser.add_argument("--lib", choices=["cuda", "c"], default="c")

args = parser.parse_args()

# process
results = hardtarget.compute_gmf(
    rx=(Path(args.drf_path), args.rx_channel),
    tx=(Path(args.drf_path), args.rx_channel),
    config=args.config,
    job={"idx": rank, "N": size},
    gmf_implementation=args.lib,
    gmf_method="fgmf",
    clobber=True,
    output=Path(args.output_path),
    start_time="2021-04-12T12:15:40",
    end_time="2021-04-12T12:16:10",
    relative_time=False,
    progress=True,
    progress_position=rank,
)

pprint(results)
