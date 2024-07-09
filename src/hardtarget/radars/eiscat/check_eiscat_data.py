from convert import loadmat, all_files
from pathlib import Path
from hardtarget.radars.eiscat.util import parse_matlab
import digital_rf as drf

RAW = "/cluster/projects/p106119-SpaceDebrisRadarCharacterization/raw"
SRC = Path(RAW) / "leo_bpark_2.0_CN-20151019-32m" / "leo_bpark_2.0_CN@32m"


if __name__ == "__main__":

    import pprint

    SAMPLE_RATE = 1000000
    FILE_SECS = 12.8

    files = list(all_files(SRC))
    n_files = len(files)

    # iterate sorted list of files
    for file in files[:1]:
        mat = loadmat(file)
        d = parse_matlab(mat, SAMPLE_RATE, FILE_SECS)
        print(f"{d['idx_file']}/{n_files}")








