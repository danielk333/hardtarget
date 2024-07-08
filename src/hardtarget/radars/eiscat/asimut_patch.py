from convert import loadmat, all_files
from pathlib import Path
import digital_rf as drf

RAW = "/cluster/projects/p106119-SpaceDebrisRadarCharacterization/raw"
SRC = Path(RAW) / "leo_bpark_2.0_CN-20151019-32m" / "leo_bpark_2.0_CN@32m"

files = list(all_files(SRC))
first = files[0]


print(first)

mat = loadmat(first)



elevation = mat["d_parbl"][0][8]
asimut = mat["d_parbl"][0][9]
print(elevation, asimut)

print(mat["d_parbl"][0])
print(mat["d_parbl"][0].dtype)



zz = mat["d_raw"][:, 0]
print(zz.shape)
