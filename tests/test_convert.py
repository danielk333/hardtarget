import pytest
from pathlib import Path
from hardtarget.utils import index_from_ts, ts_from_index
import numpy as np
import numpy.testing as npt
import hardtarget.digitalrf_wrapper as drf_wrapper
import hardtarget.analysis.utils as utils
from hardtarget.utils import ts_from_str
import h5py
import bisect

"""
test products converted to Hardtarget DRF in various ways
"""

PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
RAW = PROJECT / "raw" 
DRF = PROJECT / "drf"

def prerequisites():
    return DRF.is_dir() and RAW.is_dir()


@pytest.mark.skipif(not prerequisites() , reason="Local file is missing")
def test_pointing():

    """
    test that there is a correspondence between number of zip files in a 
    RAW Eiscat format, and the number of entries in the pointing dataset of 
    the derived Hardtarget DRF 

    Test all product in DRF directory
    """

    # testing only the first DRF
    for drf_product in list(DRF.iterdir())[:1]:

        # check that drf product has pointing
        if not (drf_product / "pointing").is_dir():
            continue     

        bad = []

        # check that pointing corresponds with raw product
        raw_product = RAW / drf_product.name
        for subfolder in raw_product.iterdir():
            if subfolder.name.endswith("information"):
                continue
            # matlab files
            bz2_files = list(sorted([file for file in subfolder.rglob("*.bz2") if file.is_file()]))
            # pointing data
            reader = drf_wrapper.DigitalMetadataReader(drf_product, "pointing")
            bounds = reader.get_bounds()
            pointing_data = list(reader.read(*bounds))

            if len(pointing_data) != len(bz2_files):
                bad.append((drf_product.name, len(bz2_files), len(pointing_data)))


        for name, n_files, n_pointing in bad:
            print(f"{name} zip files {n_files} pointing {n_pointing}")

        assert len(bad) == 0
            









