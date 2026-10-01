import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true")
    parser.addoption("--usr", action="store", help="CDSE Username")
    parser.addoption("--pwd", action="store", help="CDSE Password")
    parser.addoption("--impl", action="store", default="numpy", help="analysis implementation")
    parser.addoption(
        "--orbit-path",
        action="store",
        default=None,
        help=(
            "Path to folder with the sentinell 2 satellite high precision orbit data, "
            "if the data does not exist it is downloaded with the CDSE credentials."
            "If this is not given, the data is downloaded to a temporary directory."
        ),
    )
    parser.addoption(
        "--radar-path",
        action="store",
        help="Path to measurement of the sentinel 2 satellite, e.g ~/Eiscat/leo/EISCAT_leo_mpark_2.1u_EI@uhf_20240704_100019_278878.hdf5",
    )


@pytest.fixture(scope="session")
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture(scope="session")
def data_params(request):

    username = request.config.getoption("--usr")
    password = request.config.getoption("--pwd")
    measurement = request.config.getoption("--radar-path")
    orbit = request.config.getoption("--orbit-path")
    impl = request.config.getoption("--impl")

    if ((username and password) or orbit) and measurement:
        return (
            username,
            password,
            measurement,
            orbit,
            impl,
        )
    else:
        pytest.skip(reason="Missing CDSE Data or Username/Password or Eiscat measurement path")
