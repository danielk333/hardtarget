# # Implementation performance comparison
# ---
# A simple comparison between the c and numpy implementation, no cuda available.

import os

# Workaround to make jupyter notebook find utils
import sys
import tempfile
from pathlib import Path

from radardef import RadarDef
from radardef.types import TargetFormat

import hardtarget
from hardtarget.types.types import Job

sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "analysis"))
import utils

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "test.ini"


raw_data_dir = tempfile.TemporaryDirectory()
raw_data = utils.download_test_data(Path(raw_data_dir.name))
converted_data_path = tempfile.TemporaryDirectory()
converted_files = RadarDef().convert(raw_data, TargetFormat.H5, converted_data_path.name)

# Does not work if yappi is not installed
hardtarget.profile()


# Do computation
for impl in [hardtarget.types.Impl.c, hardtarget.types.Impl.numpy]:
    # process
    results = hardtarget.analyse(
        path=converted_files[0],
        config=config,
        job=Job(idx=0, N=1),
        implementation=impl,
        start_time=0,
        end_time=200000,
        relative_time=True,
        progress=False,
    )

    # print and clear
    stats, total = hardtarget.get_profile()
    print(f"LIB={impl}: total time = {total:.4f} [s]")
    hardtarget.print_profile(stats, total=total, max_rows=5)
    hardtarget.profile_clear()

raw_data_dir.cleanup()
converted_data_path.cleanup()
