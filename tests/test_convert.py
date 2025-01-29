import pytest
from pathlib import Path
import hardtarget.digitalrf_wrapper as drf_wrapper
import datetime as dt
from hardtarget.radars.eiscat.util import eiscat_files, eiscat_process
from hardtarget.radars.eiscat import convert
import pprint


"""
test products converted to Hardtarget DRF in various ways
"""

PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
RAW = PROJECT / "raw"
DRF = PROJECT / "drf"
SOURCE = RAW / "leo_bpark_2.1u_NO-20190606-UHF/leo_bpark_2.1u_NO@uhf"



MAX_DROPPED = 10

@pytest.mark.skipif(not SOURCE.is_dir(), reason="Local file missing")
def test_pointing():

    """
    test that there is a correspondence between number of zip files in a
    RAW Eiscat format, and the number of entries in the pointing dataset of
    the derived Hardtarget DRF

    Test all product in DRF directory
    """

    # testing only the first DRF
    # for raw_product in RAW.iterdir():

    def run(raw_product): 
        if not raw_product.is_dir():
            return
        drf_product = DRF / raw_product.relative_to(RAW)

        # check that drf product has pointing
        if not (drf_product / "pointing").is_dir():
            return

        bad = []

        # check that pointing corresponds with raw product
        raw_product = RAW / drf_product.name
        for subfolder in raw_product.iterdir():
            if subfolder.name.endswith("information"):
                continue
            print(subfolder.name)
            # matlab files
            bz2_files = list(sorted([file for file in subfolder.rglob("*.bz2") if file.is_file()]))
            # pointing data
            reader = drf_wrapper.DigitalMetadataReader(drf_product, "pointing")
            bounds = reader.get_bounds()
            pointing_data = list(reader.read(*bounds))

            # some bz2 files might have been dropped. Assume that at most 10 are dropped
            assert len(pointing_data) >= len(bz2_files) - MAX_DROPPED

            missing =  len(bz2_files) - len(pointing_data)
            if missing > 0:
                bad.append((drf_product.name, len(bz2_files), len(pointing_data), missing))

        for name, n_files, n_pointing, n_missing in bad:
            print(f"{name} zip files {n_files} pointing {n_pointing} missing {n_missing}")

    run(SOURCE)    


######################################################################################
# TEST CONVERT PROCESSING
######################################################################################

@pytest.mark.skipif(not SOURCE.is_dir(), reason="Local file missing")
def test_meta(pytestconfig):
    """
    This test prints the meta info extracted from 2 random eiscat files
    """
    if not pytestconfig.getoption("--run-slow"):
        pytest.skip("Skipped because it is marked as slow.")

    files = eiscat_files(SOURCE, start=193, count=2)
    for meta in eiscat_process(files):
        pprint.pprint(meta, sort_dicts=False)


@pytest.mark.skipif(not SOURCE.is_dir(), reason="Local file missing")
def test_convert(pytestconfig):
    if not pytestconfig.getoption("--run-slow"):
        pytest.skip("Skipped because it is marked as slow.")
    SOURCE = RAW / "leo_bpark_2.2_EI-20151027-42m/leo_bpark_2.2_EI@42m"
    DST = "/tmp"
    convert(SOURCE, DST, progress=True, start=193, count=2)
    # TODO - check that something is written to DST


@pytest.mark.skipif(not SOURCE.is_dir(), reason="Local file missing")
def test_timestamps(pytestconfig):
    """
    This tests a particular product with an issue around the given
    timestamp - visually verifying that the eiscat convert process handles
    it appropriately
    """
    if not pytestconfig.getoption("--run-slow"):
        pytest.skip("Skipped because it is marked as slow.")

    # time region
    ts_exp = 1559837299200000 / 1e6
    ts_start = ts_exp - 40
    ts_end = ts_exp + 40

    dt_start = dt.datetime.fromtimestamp(ts_start, tz=dt.timezone.utc)
    dt_end = dt.datetime.fromtimestamp(ts_end, tz=dt.timezone.utc)
    bz2_files = eiscat_files(SOURCE, start=dt_start, end=dt_end)

    for meta in eiscat_process(bz2_files):
        if meta["errors"]:
            print(meta["errors"])
