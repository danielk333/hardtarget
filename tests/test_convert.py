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



GMF = PROJECT / "gmf"
NEWGMF = PROJECT / "test/debug-2024-09-06/ingar_gmf"


def lookup(alist, spacing, start, end):
    """
    list is sorted list of regularly spaced float values
    each float f is valid in interval [f,f+spacing)
    start and end are float values, representing search interval [start, end),
    where start < end
    return indexes for elements in list whose valid interval intersects with search interval
    """
    assert start <= end
    if not alist:
        return []

    if end <= alist[0]:
        return []
    if alist[-1] + spacing <= start:
        return []

    # iterate from left
    # find index of rightmost element, where (val <= start)
    start_idx = -1
    for val in alist:
        if val <= start:
            start_idx += 1
        else:
            break
    if start_idx == -1:
        # leftmost (val > start)
        start_idx = 0

    # iterate from right
    # find index of leftmost element, where (val + spacing > end)
    n = len(alist)
    end_idx = n
    for val in alist[::-1]:
        if val + spacing > end:
            end_idx -= 1
        else:
            break
    if end_idx == n:
        # rightmost (val+spacing <= end)
        end_idx = n - 1
    return list(range(start_idx, end_idx + 1))



def test_lookup():
    

    l = [float(n) for n in range(10)]
    spacing = 1.0

    # overlap
    res = lookup(l, spacing, -1, 12.0)
    assert res[0] == 0 and res[-1] == 9

    # overlap
    res = lookup(l, spacing, -1, 10.0)
    assert res[0] == 0 and res[-1] == 9

    # overlap
    res = lookup(l, spacing, -1, 9.9)
    assert res[0] == 0 and res[-1] == 9

    # overlap
    res = lookup(l, spacing, -1, 9.0)
    assert res[0] == 0 and res[-1] == 9
    
    # not overlap
    res = lookup(l, spacing, -1, 8.9)
    assert res[0] == 0 and res[-1] == 8

    # overlap
    res = lookup(l, spacing, -0.1, 12)
    assert res[0] == 0 and res[-1] == 9
    
    # overlap
    res = lookup(l, spacing, 0, 12)
    assert res[0] == 0 and res[-1] == 9

    # overlap
    res = lookup(l, spacing, 0.9, 12)
    assert res[0] == 0 and res[-1] == 9

    # not overlap
    res = lookup(l, spacing, 1.0, 12)
    assert res[0] == 1 and res[-1] == 9

    # single hit
    res = lookup(l, spacing, 5.2, 5.8)
    assert res[0] == 5 and res[-1] == 5
    assert len(res) == 1

    # single hit
    res = lookup(l, spacing, 5.2, 5.2)
    assert res[0] == 5 and res[-1] == 5
    assert len(res) == 1

    # list with single item - hit
    res = lookup([4], spacing, 3.2, 5.8)
    assert res[0] == 0 and res[-1] == 0
    assert len(res) == 1

    # list with single item - hit
    res = lookup([4], spacing, 4.9, 5.8)
    assert res[0] == 0 and res[-1] == 0
    assert len(res) == 1

    # list with single item - miss
    res = lookup([4], spacing, 5.0, 5.8)
    assert len(res) == 0

    # list with no items
    res = lookup([], spacing, 5.0, 5.8)
    assert len(res) == 0


def test_gmf_equivalence():
    """
    testing that gmf data is time-aligned the same,
    whether it is based on old DRF or new DRF 

    GMF is analyzed using DRF

    NEW GMF is analyzed from NEW DRF, for 1 minute
    -s 2021-11-23T10:00:00 -e 2021-11-23T10:01:00
    """

    old_gmf = GMF / PRODUCT

    # need to find the correct gmf files within gmf products
    # gmf filenames are in unix time microseconds of first sample in file
    # gmf also has metadata which I can use
    files = sorted(utils.all_gmf_h5_files(str(old_gmf)))
    #with h5py.File(files[0], "r") as f:        
    #    chnl = f["experiment"].attrs["rx_channel"]
    #    ipp = f["experiment"].attrs["ipp"]
    #    n_ipp = f["processing"].attrs["n_ipp"]
    #    num_cohints_per_file = f["processing"].attrs["num_cohints_per_file"]

    # make datas
    start_ts = ts_from_str("2021-11-23T10:00:00.0")
    end_ts = ts_from_str("2021-11-23T10:01:00.0")

    def get_ts_from_path(p):
        return float (p.name.split('-')[1].split('.')[0])

    ts_list = [get_ts_from_path(f) for f in files]




    import pprint
    pprint.pprint([get_ts_from_path(p) for p in files[:10]])

    print(start_ts*1000000)
    print(end_ts*1000000)

