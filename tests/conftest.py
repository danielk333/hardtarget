import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true")
    parser.addoption("--usr", action="store", help="CDSE Username")
    parser.addoption("--pwd", action="store", help="CDSE Password")
    parser.addoption(
        "--path",
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
    measurement = request.config.getoption("--path")

    if username and password and measurement:
        return (
            username,
            password,
            measurement,
        )
    else:
        pytest.skip(reason="Missing CDSE Username/Password or Eiscat measurement path")
