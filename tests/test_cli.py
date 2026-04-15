import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import tests.utils as utils
from hardtarget.cli import inspect_analysed, inspect_raw_data, plot_analysed_data, plot_raw_data
from hardtarget.cli.cmd_analyse import AnalyseParser
from hardtarget.constants import (
    AnalysisMethod,
    DOAMethod,
    EchoSearchMethod,
    OptimizationMethod,
    TargetEstimationMethod,
)
from hardtarget.utils.h5_tools import get_analysed_h5_files

# This file cannot be randomized as it has dependencies towards eachother
pytestmark = pytest.mark.random_order(disabled=True)


@pytest.fixture(scope="module", autouse=True)
def get_data():
    # Setup
    dir = tempfile.TemporaryDirectory()
    raw_data = utils.download_test_data(Path(dir.name))
    data_dir = utils.convert_test_data(raw_data, Path(dir.name) / "data")[0]
    tmp_dir = Path(dir.name) / "tmp"
    tmp_dir.mkdir(exist_ok=False)
    # Run test
    yield data_dir, tmp_dir
    # Teardown
    dir.cleanup()


def get_analysed_path(tmp_dir_path: Path, method: str):
    return tmp_dir_path / method


def create_cfg(path: Path) -> Path:
    cfg_str = f"""
    [processing]
        n_ipp=2
        ipp_offset=0
        min_range_gate=81
        max_range_gate=138
        range_gate_step=1
        num_cohints_per_file=500
        tx_amp_limit = 0.2
    [target_estimation]
        min_acceleration=0
        max_acceleration=0
        range_gate_sub_resolution = 10
        frequency_decimation=1
        clutter_length=1500
        node_gpus=1
    [gmf]
        acceleration_steps = 1
    [dpt]
        ipp_delay_parameter=1
    [optimization]
        path = {str(get_analysed_path(path, AnalysisMethod.target_estimation.name))}
    [echo_search]
        doppler_freq_min= -30000
        doppler_freq_max = 5000
        doppler_freq_step = 1000
    [direction_of_arrival]
        elevation_limit = 0
        resolution = 10
    """
    cfg_path = path / "cfg.ini"
    f = open(cfg_path, "w")
    f.write(cfg_str)
    f.seek(0)

    return cfg_path


# ------------ Analysed data ------------


@pytest.mark.parametrize(
    "method, sub_methods",
    [
        pytest.param(
            AnalysisMethod.target_estimation,
            TargetEstimationMethod,
            marks=pytest.mark.dependency(name=AnalysisMethod.target_estimation),
        ),
        pytest.param(
            AnalysisMethod.optimize,
            OptimizationMethod,
            marks=pytest.mark.dependency(name=AnalysisMethod.optimize),
        ),
        pytest.param(
            AnalysisMethod.echo_search,
            EchoSearchMethod,
            marks=pytest.mark.dependency(name=AnalysisMethod.echo_search),
        ),
        pytest.param(
            AnalysisMethod.direction_of_arrival,
            DOAMethod,
            marks=pytest.mark.dependency(name=AnalysisMethod.direction_of_arrival),
        ),
    ],
)
def test_analysis(method, sub_methods, get_data):

    # Create parser
    analyse_parser = AnalyseParser(method, sub_methods)
    parser = analyse_parser.parser_build(argparse.ArgumentParser())
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="count", default=0)

    # Verify each sub lib
    for sub_method in sub_methods:
        data_dir, tmp_dir = get_data

        # Note that for the dependent tests to work the analysed data must be in tmp_dir/method
        cfg = create_cfg(tmp_dir)
        analysed_data_path = get_analysed_path(tmp_dir, method.name)

        # CLI request
        args_str = [
            str(data_dir),
            "--config",
            str(cfg),
            "--method",
            sub_method.name,
            "-o",
            str(analysed_data_path),
            "--relative_time",
            "--s",
            "0",
            "--e",
            "10000",
        ]
        if method == AnalysisMethod.direction_of_arrival:
            args_str.append("mu")

        # Run request
        args = parser.parse_args(args_str)
        analyse_parser.main(args)

        # Verify there is data in the output
        files = get_analysed_h5_files(analysed_data_path)

        assert len(files) > 0, "No files available after analysis"


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            AnalysisMethod.target_estimation,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.target_estimation]),
        ),
        pytest.param(
            AnalysisMethod.optimize,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.optimize]),
        ),
        pytest.param(
            AnalysisMethod.echo_search,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.echo_search]),
        ),
        pytest.param(
            AnalysisMethod.direction_of_arrival,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.direction_of_arrival]),
        ),
    ],
)
def test_plot_analysed_data(method, monkeypatch, get_data):
    # mock away plt.show
    monkeypatch.setattr(plt, "show", lambda: None)

    # get data storage
    _, tmp_dir = get_data

    analysed_dir = get_analysed_path(tmp_dir, method.name)

    parser = plot_analysed_data.parser_build(argparse.ArgumentParser())
    args = parser.parse_args([str(analysed_dir)])
    plot_analysed_data.main(args)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            AnalysisMethod.target_estimation,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.target_estimation]),
        ),
        pytest.param(
            AnalysisMethod.optimize,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.optimize]),
        ),
        pytest.param(
            AnalysisMethod.echo_search,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.echo_search]),
        ),
        pytest.param(
            AnalysisMethod.direction_of_arrival,
            marks=pytest.mark.dependency(depends=[AnalysisMethod.direction_of_arrival]),
        ),
    ],
)
def test_inspect_analysed(method, get_data):
    # get data storage
    _, tmp_dir = get_data

    analysed_dir = get_analysed_path(tmp_dir, method.name)

    # Assert no fault when running inspect
    parser = inspect_analysed.parser_build(argparse.ArgumentParser())
    args = parser.parse_args([str(analysed_dir)])
    inspect_analysed.main(args)


# ------------ Raw data ----------------


def test_plot_raw_data(monkeypatch, get_data):
    # mock away plt.show
    monkeypatch.setattr(plt, "show", lambda: None)

    # get data storage
    data_dir, _ = get_data

    parser = plot_raw_data.parser_build(argparse.ArgumentParser())
    args = parser.parse_args([str(data_dir), "--relative_time", "-s", "0", "-e", "10000"])
    plot_raw_data.main(args)


def test_inspect_raw(get_data):
    # get data storage
    data_dir, _ = get_data

    # Assert no fault when running inspect
    parser = inspect_raw_data.parser_build(argparse.ArgumentParser())
    args = parser.parse_args([str(data_dir)])
    inspect_raw_data.main(args)
