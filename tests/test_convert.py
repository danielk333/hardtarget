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



def test_gmf_equivalence():
    """
    testing that gmf data is time-aligned the same,
    whether it is based on old DRF or new DRF 

    GMF is analyzed using DRF

    NEW GMF is analyzed from NEW DRF, for 1 minute
    -s 2021-11-23T10:00:00 -e 2021-11-23T10:01:00
    """

    old_gmf = GMF / PRODUCT
    TS = ts_from_str("2021-11-23T10:00:00.0")


    def get_data_from_product(product, ts):

        # need to find the correct gmf files within gmf products
        # gmf filenames are in unix time microseconds of first sample in file
        # gmf also has metadata which I can use
        files = sorted(utils.all_gmf_h5_files(str(product)))

        # choose one specific point in time
        # find the file which covers this ts

        def get_ts_from_path(p):
            return float (p.name.split('-')[1].split('.')[0]) / 1e6

        def is_relevant(f):
            file_start_ts = get_ts_from_path(f)
            file_end_ts = file_start_ts + 2.0
            return file_start_ts <= ts and ts < file_end_ts

        relevant_files = [f for f in files if is_relevant(f)]
        print(relevant_files)
        relevant_file = relevant_files[0]

        file_start_ts = get_ts_from_path(relevant_file)
        delta = ts - file_start_ts
        print("delta", delta)



        with h5py.File(relevant_file, "r") as f:        
            #chnl = f["experiment"].attrs["rx_channel"]
            #ipp = f["experiment"].attrs["ipp"]
            #n_ipp = f["processing"].attrs["n_ipp"]
            #num_cohints_per_file = f["processing"].attrs["num_cohints_per_file"]
            sample_rate = f["experiment"].attrs["sample_rate"]
            epoch = f['epoch_unix'][()]
            print(epoch)


            print(list(f.keys()))
            print(f['sample_numbers'][:])

            idx = index_from_ts(ts, sample_rate, ts_offset_sec=epoch)
            print("idx", idx)



        # find index correspoinding to TS
        


    get_data_from_product(old_gmf, TS)





