import datetime as dt
import tempfile
from pathlib import Path

from hardtarget.constants import ConfigSubSection
from hardtarget.data_handling.configuration import extract_config_section, load_config_params
from hardtarget.echo_search.types import EchoSearchCfgParams
from hardtarget.optimization.types import OptimizeCfgParams
from hardtarget.process.utils import sample_interval_to_closest_ipp
from hardtarget.target_estimation.gmf.types import GMFCfgParams, TargetEstimationCfgParams
from hardtarget.types import Bounds, CfgParams
from hardtarget.utils.time_conversion import time_interval_to_sample_bound


def test_time_interval_to_sample_bounds():

    time_max_min_us = Bounds(1772090400000000, 1772090580000000)  # 2026-02-26T07:20:00, 2026-02-26T07:23:00

    # Start time of 2026-02-26T07:21:00
    start_time_us = 1772090460000000
    start_time_relative = 60 * 1e6
    start_time_dt = dt.datetime.strptime("2026-02-26T07:21:00", "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=dt.timezone.utc
    )

    # End time of 2026-02-26T07:21:36
    end_time_us = 1772090496000000
    end_time_relative = 96 * 1e6
    end_time_dt = dt.datetime.strptime("2026-02-26T07:21:36", "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=dt.timezone.utc
    )

    sample_rate = 1

    # Test us since epoch
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us, sample_rate=sample_rate, start_time=start_time_us, end_time=end_time_us
    )
    assert sample_bounds.start == 60
    assert sample_bounds.end == 96

    # test datetime
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us, sample_rate=sample_rate, start_time=start_time_dt, end_time=end_time_dt
    )
    assert sample_bounds.start == 60
    assert sample_bounds.end == 96

    # Test relative time
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us,
        sample_rate=sample_rate,
        start_time=int(start_time_relative),
        end_time=int(end_time_relative),
        relative_time=True,
    )
    assert sample_bounds.start == 60
    assert sample_bounds.end == 96

    # Verify unbound
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us,
        sample_rate=sample_rate,
    )
    assert sample_bounds.start == 0
    assert sample_bounds.end == 180

    assert True


def test_load_config_params():

    cfg_str = """
    [processing]
        n_ipp=3
        ipp_offset=0
        samp_offset=5
        min_range_gate=10
        max_range_gate=100
        range_gate_step=5
        num_cohints_per_file=50
        tx_amp_limit=2.0
        node_gpus=2
    """

    with tempfile.NamedTemporaryFile(mode="w+") as tmp_config:
        tmp_config.write(cfg_str)
        tmp_config.seek(0)
        tmp_config_path = tmp_config.name

        cfg = load_config_params(tmp_config_path)

        assert cfg.n_ipp == 3
        assert cfg.ipp_offset == 0
        assert cfg.samp_offset == 5
        assert cfg.min_range_gate == 10
        assert cfg.max_range_gate == 100
        assert cfg.range_gate_step == 5
        assert cfg.num_cohints_per_file == 50
        assert cfg.tx_amp_limit == 2.0
        assert cfg.node_gpus == 2


def test_extract_config_section():
    cfg_str = """
    [processing]
        n_ipp=3
        ipp_offset=0
        samp_offset=5
        min_range_gate=10
        max_range_gate=100
        range_gate_step=5
        num_cohints_per_file=50
        tx_amp_limit=2.0
        node_gpus=2
    [target_estimation]
        min_acceleration=30
        max_acceleration=93
        range_gate_sub_resolution = 10
        frequency_decimation=1
        clutter_length=1500
        node_gpus=1
    [gmf]
        acceleration_steps = 10
    [dpt]
        ipp_delay_parameter=3
    [optimization]
        path = "example/path"
    [echo_search]
        doppler_freq_min= -30000
        doppler_freq_max = 5000
        doppler_freq_step = 1000
    """

    with tempfile.NamedTemporaryFile(mode="w+") as tmp_config:
        tmp_config.write(cfg_str)
        tmp_config.seek(0)
        tmp_config_path = Path(tmp_config.name)

        # --- Processing section ---
        processing = extract_config_section(tmp_config_path, ConfigSubSection.PROCCESSING, CfgParams)
        cfg = CfgParams(**processing)
        assert cfg.n_ipp == 3
        assert cfg.ipp_offset == 0
        assert cfg.samp_offset == 5
        assert cfg.min_range_gate == 10
        assert cfg.max_range_gate == 100
        assert cfg.range_gate_step == 5
        assert cfg.num_cohints_per_file == 50
        assert cfg.tx_amp_limit == 2.0
        assert cfg.node_gpus == 2

        # --- Target estimation section ---
        target_estimation_params = extract_config_section(
            tmp_config_path, ConfigSubSection.TARGET_ESTIMATION, TargetEstimationCfgParams
        )
        assert target_estimation_params["min_acceleration"] == 30
        assert target_estimation_params["max_acceleration"] == 93
        assert target_estimation_params["range_gate_sub_resolution"] == 10
        assert target_estimation_params["frequency_decimation"] == 1
        assert target_estimation_params["clutter_length"] == 1500
        assert target_estimation_params["node_gpus"] == 1

        # --- GMF section ---
        gmf_params = extract_config_section(tmp_config_path, ConfigSubSection.GMF, GMFCfgParams)
        assert gmf_params["acceleration_steps"] == 10

        # --- Optimization section ---
        optimize_params = extract_config_section(
            tmp_config_path, ConfigSubSection.OPTIMIZATION, OptimizeCfgParams
        )
        assert optimize_params["path"] == "example/path"

        # --- Echo search section ---
        echo_search_params = extract_config_section(
            tmp_config_path, ConfigSubSection.ECHO_SEARCH, EchoSearchCfgParams
        )
        assert echo_search_params["doppler_freq_min"] == -30000
        assert echo_search_params["doppler_freq_max"] == 5000
        assert echo_search_params["doppler_freq_step"] == 1000


def test_sample_interval_to_closest_ipp():
    """verify that it rounds to closest ipp"""

    sample_bounds = Bounds(60, 446)
    ipp_samps = 55

    bounds = sample_interval_to_closest_ipp(sample_bounds, ipp_samps)

    assert bounds.start == 55
    assert bounds.end == 440
