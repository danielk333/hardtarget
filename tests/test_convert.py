import pytest
from pathlib import Path
import hardtarget.digitalrf_wrapper as drf_wrapper
import datetime as dt
from hardtarget.radars.eiscat.convert import loadmat, index_of_filestart, load_expconfig, expinfo_split
from hardtarget.radars.eiscat.convert import to_i2x16

"""
test products converted to Hardtarget DRF in various ways
"""

PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
RAW = PROJECT / "raw"
DRF = PROJECT / "drf"


def prerequisites():
    return DRF.is_dir() and RAW.is_dir()


MAX_DROPPED = 10

@pytest.mark.skipif(not prerequisites(), reason="Local file is missing")
def test_pointing():

    """
    test that there is a correspondence between number of zip files in a
    RAW Eiscat format, and the number of entries in the pointing dataset of
    the derived Hardtarget DRF

    Test all product in DRF directory
    """

    # testing only the first DRF
    for raw_product in RAW.iterdir():

        if not raw_product.is_dir():
            continue

        drf_product = DRF / raw_product.relative_to(RAW)

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

            # some bz2 files might have been dropped. Assume that at most 10 are dropped
            assert len(pointing_data) >= len(bz2_files) - MAX_DROPPED

            missing =  len(bz2_files) - len(pointing_data)
            if missing > 0:
                bad.append((drf_product.name, len(bz2_files), len(pointing_data), missing))

        for name, n_files, n_pointing, n_missing in bad:
            print(f"{name} zip files {n_files} pointing {n_pointing} missing {n_missing}")




######################################################################################
# TEST CONVERT LOOP
######################################################################################

SOURCE = RAW / "leo_bpark_2.1u_NO-20190606-UHF/leo_bpark_2.1u_NO@uhf"



@pytest.mark.skipif(not SOURCE.is_dir(), reason="Local file missing")

def test_timestamps(pytestconfig):
    if not pytestconfig.getoption("--run-slow"):
        pytest.skip("Skipped because it is marked as slow.")

    def beginning_of_year(_dt):
        return _dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    def get_seconds_since_year_start(_dt):
        _dt_year_start = beginning_of_year(_dt)
        return int((_dt - _dt_year_start).total_seconds())

    # time region
    exp = 1559837299200000 / 1e6

    dt_exp = dt.datetime.fromtimestamp(exp, tz=dt.timezone.utc)

    # calculate time offset in seconds since start of year - used in filenames
    target_offset = get_seconds_since_year_start(dt_exp)

    # find all bz2 files which are close in time
    bz2_files = sorted([file for file in SOURCE.rglob("*.bz2") if file.is_file()])

    def select(file):
        # get file offset
        offset = int(file.name.split(".")[0])
        return abs(target_offset - offset) < 40

    bz2_files = [f for f in bz2_files if select(f)]

    mat_first = loadmat(str(bz2_files[0]))
    mat_last = loadmat(str(bz2_files[-1]))

    host, expname, expvers, owner = expinfo_split(str(mat_last["d_ExpInfo"][0]))
    cfg = load_expconfig(expname)
    cfv = cfg[expvers]
    sample_rate = float(cfv.get("sample_rate"))
    file_secs = float(cfv.get("file_secs"))
    samples_per_file = int(file_secs * sample_rate)
    # chnl = cfv.get("rx_channel", "tbd")
    # radar_frequency = float(mat_last["d_parbl"][0][PARBL_RADAR_FREQUENCY])

    idx_start = index_of_filestart(mat_first, sample_rate, file_secs)
    # idx_end = index_of_filestart(mat_last, sample_rate, file_secs) + samples_per_file
    # n_files = len(bz2_files)

    def zeropad(n_pad, file):
        print(f"zeropad {n_pad} bytes")

    def write(zz, file):
        print(f"write {len(zz)} bytes")

    def drop(zz, file):
        print(f"drop {len(zz)} bytes")

    print("start")

    # index of next write
    idx_write = idx_start

    for file in bz2_files:

        #################
        # initialise
        #################

        mat = loadmat(str(file))
        process_ok = True
        chunk_idx_start = index_of_filestart(mat, sample_rate, file_secs)
        zz = to_i2x16(mat["d_raw"][:, 0])
        n_samples = len(zz)

        #################
        # check
        #################

        # check length of data
        # conservative - must be exact
        if n_samples != samples_per_file:
            process_ok = False

        # check that we are not writing old data
        if chunk_idx_start < idx_write:
            process_ok = False

        # check that idx_start is aligned with logical file boundaries
        remainder = (chunk_idx_start - idx_start) % samples_per_file
        if remainder != 0.0:
            # illegal index
            process_ok = False

        # check if filenames are consistent with read index
        # filenames are given by end_index (first index of next chunk)
        idx_last = chunk_idx_start + samples_per_file
        ts_last = idx_last / 1e6
        dt_last = dt.datetime.fromtimestamp(ts_last, tz=dt.timezone.utc)
        offset = get_seconds_since_year_start(dt_last)
        if offset != int(file.name.split(".")[0]):
            # filename inconsistency
            process_ok = False

        print(f"exp {idx_write/1e6} got {chunk_idx_start/1e6} exp {offset} got {file.name}")

        #################
        # process
        #################

        if process_ok:
            # if necessary, zero pad stream up to idx_write
            n_pad = chunk_idx_start - idx_write
            if n_pad > 0:
                zeropad(n_pad, file)
            # write chunk to stream
            write(zz, file)
        else:
            # drop
            drop(zz, file)
            # do not increment idx_write
            continue

        # increment idx_write
        idx_write = chunk_idx_start + samples_per_file
