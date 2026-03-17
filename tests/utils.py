import urllib.request
from pathlib import Path

from radardef import RadarDef


def download_test_data(path: Path) -> Path:

    download_location = path / "MUI.000000.000000"
    if not download_location.is_file():
        url = "https://cloud.irf.se/public.php/dav/files/pXR6iYARobLxd2f/?accept=zip"
        urllib.request.urlretrieve(url, download_location)
    return download_location


def convert_test_data(path: Path, dst: Path) -> list[Path]:
    radars = RadarDef()
    source_format = radars.get_source_format(path)
    target_formats = radars.available_target_formats(source_format)
    converted_files = RadarDef().convert(path, target_formats[0], str(dst))
    assert converted_files is not None, "No available files after conversion"
    return converted_files
