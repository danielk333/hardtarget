"""
This test is verifying the hardtarget functionality by comparing a precision orbit satellite (SENTINEL-2B)
position and hardtargets range and velocity estimation from the eiscat uhf (Tromso) measurement of the object.

The object is noticed in the EISCAT_leo_mpark_2.1u_EI@uhf_20240704_100019_278878.hdf5 at 24-07-04T10:21:16
"""

import datetime as dt
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import radardef
from pyant.plotting import gain_heatmap
from radardef.types import BeamType, EiscatUHFLocation
from spacecoords import interpolation, linalg, spherical

from hardtarget import load_analysed_data, target_estimation
from hardtarget.constants import TargetEstimationMethod
from hardtarget.target_estimation.dpt.types import DPTCfgParams
from hardtarget.target_estimation.gmf.types import GMFCfgParams
from hardtarget.target_estimation.types import MFOutArgs
from hardtarget.types import CfgParams, ExpDef, ProParams

from .utils import cdse
from .utils.dt_standard import str_to_dt

# ------- USER VARS ----------
COPERNICUS_USR = "TODO"
COPERNICUS_PWD = "TODO"
EISCAT_UHF_MEASUREMENT = Path("/Eiscat/leo/EISCAT_leo_mpark_2.1u_EI@uhf_20240704_100019_278878.hdf5")
# ----------------------------


def get_eiscat_data(dt_start: dt.datetime, dt_end: dt.datetime) -> Path:
    # TODO: When possible fix so this actually searches or gets data from the Eiscat database
    return EISCAT_UHF_MEASUREMENT


# Process specific configurations
gmf_cfg = GMFCfgParams(
    n_ipp=1,
    ipp_offset=0,
    samp_offset=3,
    min_range_gate=4000,
    max_range_gate=8000,
    min_acceleration=0,
    max_acceleration=0,
    range_gate_step=1,
    frequency_decimation=1,
    num_cohints_per_file=10,
    node_gpus=1,
    acceleration_steps=1,
)

dpt_cfg = DPTCfgParams(
    n_ipp=2,
    ipp_offset=0,
    samp_offset=3,
    min_range_gate=4000,
    max_range_gate=8000,
    min_acceleration=0,
    max_acceleration=0,
    range_gate_step=1,
    frequency_decimation=1,
    num_cohints_per_file=10,
    node_gpus=1,
    ipp_delay_parameter=1,
)


# Verify precision orbit estimation for different methods
@pytest.mark.skipif(COPERNICUS_USR == "TODO", reason="Copernicus username missing")
@pytest.mark.skipif(COPERNICUS_PWD == "TODO", reason="Copernicus password missing")
@pytest.mark.skipif(not EISCAT_UHF_MEASUREMENT.exists(), reason="Local file is missing")
@pytest.mark.parametrize(
    "params", [(TargetEstimationMethod.fgmf, gmf_cfg), (TargetEstimationMethod.fdpt, dpt_cfg)]
)
def test_verify_analysis_orbit_data(params: tuple[TargetEstimationMethod, CfgParams]):

    method_lib, cfg = params

    # Temp dir to store precision orbit data
    tmp_dir = tempfile.TemporaryDirectory()

    # Time object is noticed in eiscat data
    start_time = str_to_dt("2024-07-04T10:21:15.500")
    end_time = str_to_dt("2024-07-04T10:21:20.020")

    # get precision orbit data to interpolate
    data_id = cdse.get_orbit_data_id(start_time, end_time)
    access_token = cdse.generate_token(COPERNICUS_USR, COPERNICUS_PWD)
    orbit_data_path = cdse.download_orbit_data(
        data_id=data_id,
        access_token=access_token,
        output_dir=Path(tmp_dir.name) / "orbit_data",
    )
    orbit_data = cdse.extract_eof_data_block(
        orbit_data_path, start_time - dt.timedelta(hours=1), end_time + dt.timedelta(hours=1)
    )
    t, pos = zip(*orbit_data)

    t = np.array(
        [str_to_dt(x).timestamp() for x in t],
        dtype=np.float64,
    )

    # Interpolate satellite position to get the full orbit
    pos = np.asarray(pos, dtype=np.float64).T
    interpolated_satellite_pos = interpolation.Legendre8(states=pos, t=t)

    # Measurement source station
    radar_station = radardef.EiscatUHF(location=EiscatUHFLocation.TROMSO, beam_type=BeamType.CASSEGRAIN)

    # Analyse data
    result = target_estimation(
        data=get_eiscat_data(start_time, end_time),
        config=cfg,
        output=Path(tmp_dir.name) / "analysed",
        start_time=start_time,
        end_time=end_time,
        relative_time=False,
        progress=False,
        method_lib=method_lib,
    )

    assert result["dir"] is not None

    # Load results
    load_ret: tuple[MFOutArgs, ExpDef, GMFCfgParams, ProParams]
    load_ret = list(load_analysed_data(result["dir"]))[0]
    out, exp, cfg, pro = load_ret

    # Get satellite position over the analysed interval
    t_analysed = np.arange(
        start=start_time.timestamp(), stop=end_time.timestamp(), step=cfg.n_ipp * (exp.t_ipp_usec * 1e-6)
    )
    satellite_orbit = interpolated_satellite_pos.get_state(t_analysed)

    # Get local enu coordinates relative to the radar station
    satellite_enu = radar_station.enu(satellite_orbit)

    # Calculate range and velocity relative to the radar station
    r_rel, v_rel = generate_measurements(
        satellite_orbit,
        satellite_enu,
        satellite_enu,
    )

    # Extract indexes where an object is present
    r_inds = np.argmax(out.snr, axis=1)
    coh_inds = np.arange(out.vals.shape[0])
    snr = out.snr[coh_inds, r_inds]
    snrdb = 10 * np.log10(snr)
    inds = snrdb > 15.0

    # Calculate delta range (real vs estimated)
    dr = np.abs(r_rel[inds] - out.r_vec[inds])
    dr_limit = 500

    # Calculate delta velocity (real vs estimated)
    dv = np.abs(v_rel[inds] - out.v_vec[inds])
    dv_limit = 12

    # For debugging, if True debug plots will be shown
    debug = False
    if debug:
        # --- Plot estimation vs real range/velocity ---
        fix, ax = plt.subplots(2, 2)

        # Real vs estimated range
        ax[0, 0].plot(t_analysed, (r_rel * 0.5) / 1000, marker=".", color="g", label="Real range")
        ax[0, 0].plot(
            t_analysed[inds],
            (out.r_vec[inds] * 0.5) / 1000,
            marker=".",
            ls="none",
            color="r",
            label="Estimated range",
        )
        ax[0, 0].set_xlabel("Time [s]")
        ax[0, 0].set_ylabel("range [km]")
        ax[0, 0].set_title("Real vs Estimated Range")
        ax[0, 0].legend()

        # Range delta
        ax[0, 1].plot(t_analysed[inds], dr / 1000, marker=".", ls="none", label="|r_delta|")
        ax[0, 1].axhline(np.mean(dr) / 1000, linestyle="--", color="g", label="mean |r_delta|")
        ax[0, 1].axhline(np.abs(dr_limit) / 1000, linestyle="--", color="r", label="limit")
        ax[0, 1].set_xlabel("Time [s]")
        ax[0, 1].set_ylabel("Delta range [km]")
        ax[0, 1].legend()

        # Real vs estimated velocity
        ax[1, 0].plot(t_analysed[inds], v_rel[inds], marker=".", color="g", label="Real vel")
        ax[1, 0].plot(
            t_analysed[inds], out.v_vec[inds], marker=".", color="r", ls="none", label="Estimated vel"
        )
        ax[1, 0].set_xlabel("Time [s]")
        ax[1, 0].set_ylabel("range rate [m/s]")
        ax[1, 0].set_title("Real vs estimated velocity")
        ax[1, 0].legend()

        # velocity delta
        ax[1, 1].plot(t_analysed[inds], dv, marker=".", ls="none", label="|v_delta|")
        ax[1, 1].axhline(np.mean(dv), linestyle="--", color="g", label="mean |v_delta|")
        ax[1, 1].axhline(np.abs(dv_limit), linestyle="--", color="r", label="limit")
        ax[1, 1].set_xlabel("Time [s]")
        ax[1, 1].set_ylabel("Delta velocity [m/s]")
        ax[1, 1].legend()

        # --- Satellite orbit vs radar position ---

        #  Orbit vs position TODO: radar pointing in wrong direction in plot
        pointing_cart = spherical.sph_to_cart(
            np.array([out.pointing_vec[0, 0], out.pointing_vec[0, 1], 1]),
            degrees=True,
        ).round(decimals=8)
        satellite_cart = satellite_enu[:3, :] / np.linalg.norm(satellite_enu[:3, :], axis=0)
        off_axis_angle = linalg.vector_angle(
            satellite_cart,
            pointing_cart,
            degrees=True,
        )

        pointing_cart *= 4000e3
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 2, 1, projection="3d")
        ax.plot(pos[0, :], pos[1, :], pos[2, :], "-b")
        ax.plot(radar_station.ecef[0], radar_station.ecef[1], radar_station.ecef[2], ".g")
        ax.plot(
            [radar_station.ecef[0], radar_station.ecef[0] + pointing_cart[0]],
            [radar_station.ecef[1], radar_station.ecef[1] + pointing_cart[1]],
            [radar_station.ecef[2], radar_station.ecef[2] + pointing_cart[2]],
            "-r",
        )

        # flat distance plot over a larger timspan
        t_long = np.arange(
            start=(start_time - dt.timedelta(minutes=60)).timestamp(),
            stop=(end_time + dt.timedelta(minutes=60)).timestamp(),
            step=cfg.n_ipp * (exp.t_ipp_usec * 1e-6),
        )
        satellite_long_orbit = interpolated_satellite_pos.get_state(t_long)

        r_long, v_long = generate_measurements(
            satellite_long_orbit,
            radar_station.enu(satellite_long_orbit),
            radar_station.enu(satellite_long_orbit),
        )
        ax = fig.add_subplot(2, 2, 2)
        ax.plot(t_long, (r_long / 1000) * 0.5, color="g", label="Satelite range from radar")
        ax.axvline(t_analysed.min(), linestyle="--", color="c", label="Analys start")
        ax.axvline(t_analysed.max(), linestyle="--", color="c", label="Analys end")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Range[m]")
        ax.legend()

        # off-axis angle plot
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(t_analysed, off_axis_angle, color="g", label="Satelite angle from radar pointing")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Off-axis angle [deg]")
        ax.legend()

        # Gain heatmap vs satellite position
        ax = fig.add_subplot(2, 2, 4)
        gain_heatmap(radar_station.beam, radar_station.beam_parameters, ax=ax, min_elevation=87.0)
        ax.plot(satellite_cart[0, :], satellite_cart[1, :], label="satellite path over gain")
        ax.plot(
            satellite_cart[0, inds],
            satellite_cart[1, inds],
            color="g",
            marker=".",
            ls="none",
            label="Positions detected",
        )

        plt.show()

    tmp_dir.cleanup()

    assert np.mean(dr) < dr_limit, (
        f"Estimated range is far off the real range! limit: {dr_limit} [m] > delta: {np.mean(dr)} [m]"
    )
    assert np.mean(dv) < dv_limit, (
        f"Estimated velocity is far off the real velocity! limit: {dv_limit} [m/s] > delta: {np.mean(dv)} [m/s]"
    )


def generate_measurements(ecefs, rx_enu, tx_enu):

    tx_range = np.linalg.norm(tx_enu, axis=0)
    rx_range = np.linalg.norm(rx_enu, axis=0)
    r_sim = tx_range + rx_range
    v_tx = -np.sum(tx_enu[:3, :] * tx_enu[3:, :], axis=0) / tx_range
    v_rx = -np.sum(rx_enu[:3, :] * rx_enu[3:, :], axis=0) / rx_range
    v_sim = v_tx + v_rx

    return r_sim, v_sim


def generate_measurements_alt(ecefs, rx_ecef, tx_ecef):
    index_tuple = (slice(None),) + tuple(None for x in range(len(ecefs.shape) - 1))

    r_tx = tx_ecef[index_tuple] - ecefs[:3, :, ...]
    r_rx = rx_ecef[index_tuple] - ecefs[:3, :, ...]

    tx_range = np.linalg.norm(r_tx, axis=0)
    rx_range = np.linalg.norm(r_rx, axis=0)
    r_sim = tx_range + rx_range
    v_tx = -np.sum(r_tx[:3, :] * ecefs[3:, :], axis=0) / tx_range
    v_rx = -np.sum(r_rx[:3, :] * ecefs[3:, :], axis=0) / rx_range
    v_sim = v_tx + v_rx

    return r_sim, v_sim

    return r_sim, v_sim
