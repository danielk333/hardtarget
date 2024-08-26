import pytest
from hardtarget.radars.eiscat import convert
from hardtarget.utils import index_from_ts, ts_from_index
from hardtarget.analysis.utils import load_metadata, load_pointing_data
import numpy as np
import numpy.testing as npt
import hardtarget.digitalrf_wrapper as drf_wrapper
from pathlib import Path

PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
SRC = PROJECT / "raw/leo_bpark_2.1u_NO-20220408-UHF/leo_bpark_2.1u_NO@uhf"
CFG = Path("/cluster/home/inar/Dev/Git/hardtarget/examples/cfg/test.ini")
DST = Path("/cluster/home/inar/Data/hardtarget")

def prerequisites():
    return SRC.exists() and CFG.exists() and DST.exists()


@pytest.mark.skipif(not prerequisites() , reason="Local file is missing")
def test_convert():
    
    # convert
    result = convert(SRC, DST, name="leo_bpark_2.1u_NO-20220408-UHF", progress=True)
    print("done")
    print(result)
