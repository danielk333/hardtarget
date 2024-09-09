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
NEWDRF = PROJECT / "newdrf"
PRODUCT = "leo_bpark_2.1u_NO-20211123-UHF"


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

    for drf_product in DRF.iterdir():

        # check that drf product has pointing
        if not (drf_product / "pointing").is_dir():
            continue     

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
            assert len(pointing_data) == len(bz2_files)



def test_drf_equivalence():
    """
    testing equivalence between old and new DRF
    - bounds are not the same, but data must be
    where bounds are overlapping  
    """

    old_drf = DRF / PRODUCT
    new_drf = NEWDRF / PRODUCT

    old_reader = drf_wrapper.DigitalRFReader(old_drf, "uhf")
    new_reader = drf_wrapper.DigitalRFReader(new_drf, "uhf")

    old_bounds = old_reader.get_bounds()
    new_bounds = new_reader.get_bounds()

    _start = max(old_bounds[0], new_bounds[0])
    _end = min(old_bounds[1], new_bounds[1])

    # check beginning
    old_data = list(old_reader.read(_start, _start + 10))
    new_data = list(new_reader.read(_start, _start + 10))
    npt.assert_equal(old_data, new_data)

    # check end
    old_data = list(old_reader.read(_end - 10, _end))
    new_data = list(new_reader.read(_end - 10, _end))
    npt.assert_equal(old_data, new_data)







