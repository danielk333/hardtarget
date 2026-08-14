# # Implementation performance comparison
# ---
# A simple comparison between the c and numpy implementation, no cuda available.

import os
import sys

# Workaround to make jupyter notebook find utils
import tempfile
from pathlib import Path

import hardtarget
from hardtarget.constants import TargetEstimationMethod

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "test.ini"


tmp_dir = tempfile.TemporaryDirectory()
raw_path = Path(tmp_dir.name) / "raw"
raw_path.mkdir(parents=True, exist_ok=True)
converted_path = Path(tmp_dir.name) / "converted"
raw_data = utils.download_test_data(raw_path)
data = utils.convert_test_data(raw_data, converted_path)[0]

# Does not work if yappi is not installed
hardtarget.profile()


# Do computation
for impl in [hardtarget.types.Impl.c, hardtarget.types.Impl.numpy]:
    # process
    results = hardtarget.target_estimation(
        data=data,
        config=config,
        method_lib=TargetEstimationMethod.fgmf,
        implementation=impl,
        start_time=0,
        end_time=2,
        relative_time=True,
        progress=False,
    )

    # print and clear
    stats, total = hardtarget.get_profile()
    print(f"LIB={impl}: total time = {total:.4f} [s]")
    hardtarget.print_profile(stats, total=total, max_rows=5)
    hardtarget.profile_clear()

tmp_dir.cleanup()
