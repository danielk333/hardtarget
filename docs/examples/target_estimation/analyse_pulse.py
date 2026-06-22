# # Analyse individual radar pulse
# ---
# How to extract and analyse an individual radar pulse

import argparse
import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import radardef
import requests
import scipy.signal
from matplotlib import pyplot as plt
from radardef.types import BeamType, EiscatUHFLocation
from scipy import constants
from scipy.fft import fft, fftfreq, fftshift
from spacecoords import interpolation
from tqdm import tqdm

from hardtarget import plotting
from hardtarget.data_simulation.tx_model import match_pulse_code, tx_modulation_model, tx_signal_model
from hardtarget.plotting.raw_data_plots import extract_requested_range_gates
from hardtarget.process.utils import sample_interval_to_closest_ipp
from hardtarget.target_estimation.gmf.gmf_numpy import dtft_solve
from hardtarget.target_estimation.gmf.types import GMFCfgParams
from hardtarget.types import Bounds
from hardtarget.utils.time_conversion import time_interval_to_sample_bound, ts_from_str

parser = argparse.ArgumentParser()
parser.add_argument("data_file", type=Path)
parser.add_argument("orb_file", type=Path)
parser.add_argument("--offset", type=int, default=0)
args = parser.parse_args()


def generate_measurements(ecefs, rx_enu, tx_enu):

    tx_range = np.linalg.norm(tx_enu, axis=0)
    rx_range = np.linalg.norm(rx_enu, axis=0)
    r_sim = tx_range + rx_range
    v_tx = -np.sum(tx_enu[:3, :] * tx_enu[3:, :], axis=0) / tx_range
    v_rx = -np.sum(rx_enu[:3, :] * rx_enu[3:, :], axis=0) / rx_range
    v_sim = v_tx + v_rx

    return r_sim, v_sim


def search_for_sentinel_data(
    dt_start: dt.datetime,
    dt_end: dt.datetime,
    collection: str = "SENTINEL-2",
    object: str = "S2B",
    product_catalogue: str = "AUX_POEORB",
):
    query = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=((Collection/Name eq '{collection}') and (ContentDate/Start gt {dt_start.strftime(dt_format)}Z) and (ContentDate/Start lt {dt_end.strftime(dt_format)}Z) and ((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name eq 'productType' and i0/Value eq '{product_catalogue}'))))&$orderby=ContentDate/Start&$top=10"
    json_res = requests.get(query).json()

    value = json_res["value"]

    if not value:
        raise Exception(f"No data for {collection} found between: {dt_start} and {dt_end}")
    else:
        print(f"Found {len(value)} items")

    return value


def get_orbit_data_id(dt_start: dt.datetime, dt_end: dt.datetime):
    data = search_for_sentinel_data(dt_start - dt.timedelta(days=1), dt_end, "SENTINEL-2", "AUX_POEORB")
    return data[1]["Id"]


def extract_eof_data_block(data_path: Path, dt_start: dt.datetime, dt_end: dt.datetime):
    """

    Returns:
        list of tuples, each tuple is the timepoint with a list of x,y,z,vx.vy,vz

    """

    tree = ET.parse(str(data_path))
    root = tree.getroot()
    data_block = root.find("Data_Block")

    def data_structure(x: ET.Element):
        return [
            x.find("X").text,
            x.find("Y").text,
            x.find("Z").text,
            x.find("VX").text,
            x.find("VY").text,
            x.find("VZ").text,
        ]

    data = [
        (x.find("UTC").text[4:], data_structure(x))
        for x in data_block.findall(".//OSV")
        if time_within_block(x.find("UTC").text[4:], dt_start, dt_end)
    ]

    if not data:
        raise Exception(
            f"No data available between {dt_start.strftime(dt_format)} and {dt_end.strftime(dt_format)}, try a larger timespan"
        )

    return data


def time_within_block(dt_curr: str | dt.datetime, dt_start: dt.datetime, dt_end: dt.datetime) -> bool:

    if isinstance(dt_curr, str):
        dt_curr = str_to_dt(dt_curr)
    return dt_curr >= dt_start and dt_curr <= dt_end


def generate_token(username: str, password: str) -> str:

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }

    response = requests.post(
        url,
        headers=headers,
        data=data,
    ).json()

    if "access_token" in response:
        return response["access_token"]
    else:
        raise Exception("Bad credentials, no access token generated")


dt_format = "%Y-%m-%dT%H:%M:%S.%f"


def str_to_dt(dt_str: str):
    return dt.datetime.strptime(dt_str, dt_format).replace(tzinfo=dt.timezone.utc)


def download_orbit_data(data_id: str, access_token: str, output_dir: Path) -> Path:
    """
    https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Satellite_constellation

    Args:
        dt_start: start of data
        dt_end: end of data
    Returns:
        Orbit data within dt_start and dt_end format (timepoint, [x y z vx vy vz] (m, m/s))
    """

    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({data_id})/$value"

    headers = {"Authorization": f"Bearer {access_token}"}

    # Create a session and update headers
    session = requests.Session()
    session.headers.update(headers)

    # Perform the GET request
    response = session.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        data_path = output_dir / "data.eof"
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(str(data_path), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        return data_path
    else:
        print(f"GET response: {response.text}")
        raise Exception(f"Failed to download orbit data. Status code: {response.status_code} ")


cfg = GMFCfgParams(
    n_ipp=1,
    ipp_offset=0,
    samp_offset=args.offset,
    min_range_gate=5500,
    max_range_gate=7700,
    min_acceleration=0,
    max_acceleration=0,
    range_gate_step=1,
    range_gate_sub_resolution=10,
    frequency_decimation=1,
    num_cohints_per_file=2,
    node_gpus=1,
    acceleration_steps=1,
)
# ##Prerequisites
# ---

# Data to analyse

# Time object is noticed in eiscat data
start_time_str = "2024-07-04T10:21:17.500"
end_time_str = "2024-07-04T10:21:20.000"
start_time_dt = str_to_dt(start_time_str)
end_time_dt = str_to_dt(end_time_str)

# get precision orbit data to interpolate
data_id = get_orbit_data_id(start_time_dt, end_time_dt)

orbit_data = extract_eof_data_block(
    args.orb_file, start_time_dt - dt.timedelta(hours=1), end_time_dt + dt.timedelta(hours=1)
)
t, pos = zip(*orbit_data)

t = np.array(
    [str_to_dt(x).timestamp() for x in t],
    dtype=np.float64,
)

# interpolate satellite position to get the full orbit
pos = np.asarray(pos, dtype=np.float64).T
interpolated_satellite_pos = interpolation.Legendre8(states=pos, t=t)

# Measurement source station
radar_station = radardef.EiscatUHF(
    location=EiscatUHFLocation.TROMSO,
    beam_type=BeamType.CASSEGRAIN,
)

reader = radar_station.load_data(args.data_file)
assert reader is not None

# ## Plot
# ---
# So what is seen here is the power for each rx sample for each ipp on the logarithmic scale, for a specific
# set of ipps

start_time = int(ts_from_str(start_time_str) * 1e6)
end_time = int(ts_from_str(end_time_str) * 1e6)

request_bounds = time_interval_to_sample_bound(
    time_bounds=Bounds(int(reader.epoch_bounds.ts_start_usec), int(reader.epoch_bounds.ts_end_usec)),
    start_time=start_time,
    end_time=end_time,
    sample_rate=reader.exp_def.sample_rate,
    relative_time=False,
)

samp_bounds = sample_interval_to_closest_ipp(request_bounds, reader.exp_def.ipp_samps)

start_ipp_num = request_bounds.start // reader.exp_def.ipp_samps

# extract data within bounds
n_samp = samp_bounds.end - samp_bounds.start
data_vec = reader.read(
    channel=reader.exp_def.rx_channels,
    start_sample=samp_bounds.start + cfg.samp_offset,
    vector_length=n_samp,
)

if data_vec.ndim > 1:
    data_vec = np.sum(data_vec, axis=0)


# define experiment tx and rx intervals
def usec_to_sample(t_usec: int) -> int:
    return int(t_usec / reader.exp_def.t_samp_usec)


t_rx_start_samp = usec_to_sample(reader.exp_def.t_rx_start_usec)
t_rx_end_samp = usec_to_sample(reader.exp_def.t_rx_end_usec)
t_tx_start_samp = usec_to_sample(reader.exp_def.t_tx_start_usec)
t_tx_end_samp = usec_to_sample(reader.exp_def.t_tx_end_usec)
t_cal_on_samp = (
    usec_to_sample(int(reader.exp_def.t_cal_on_usec)) if reader.exp_def.t_cal_on_usec is not None else 0
)
t_cal_off_samp = (
    usec_to_sample(int(reader.exp_def.t_cal_off_usec)) if reader.exp_def.t_cal_off_usec is not None else 0
)

range_t = t_tx_start_samp / reader.exp_def.sample_rate
samp_vec = np.arange(reader.exp_def.ipp_samps)
rt_vec = np.arange(t_rx_end_samp - t_rx_start_samp) * reader.exp_def.t_samp_usec - range_t

mat_shape = (data_vec.size // reader.exp_def.ipp_samps, reader.exp_def.ipp_samps)
data_ipp_vec_full = data_vec.reshape(mat_shape).T

il0_rg0, il0_rg1 = extract_requested_range_gates(
    cfg.min_range_gate, cfg.max_range_gate, "sample", reader.exp_def
)
data_ipp_vec = data_ipp_vec_full[il0_rg0:il0_rg1, :]


# Get satellite position over the analysed interval
t_analysed = np.arange(
    start=start_time_dt.timestamp(),
    stop=end_time_dt.timestamp(),
    step=cfg.n_ipp * (reader.exp_def.t_ipp_usec * 1e-6),
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
time_correction = 0.5 * r_rel / scipy.constants.c
satellite_orbit = interpolated_satellite_pos.get_state(t_analysed + time_correction)
satellite_enu = radar_station.enu(satellite_orbit)
r_rel, v_rel = generate_measurements(
    satellite_orbit,
    satellite_enu,
    satellite_enu,
)

ipp_index = 13
# ipp_index = 1

print(r_rel[ipp_index], v_rel[ipp_index])
r_true, v_true = r_rel[ipp_index], v_rel[ipp_index]

signal = data_ipp_vec_full[:, ipp_index]
exp_params = reader.exp_def.copy(radar_frequency=927.200)


def usec_to_samp(usec: int | float) -> int:
    return int(usec / exp_params.t_samp_usec)


rx_start_samp = usec_to_samp(exp_params.t_rx_start_usec)
rx_end_samp = usec_to_samp(exp_params.t_rx_end_usec)
tx_start_samp = usec_to_samp(exp_params.t_tx_start_usec)
tx_end_samp = usec_to_samp(exp_params.t_tx_end_usec)
tx_pulse_samps = tx_end_samp - tx_start_samp

# Sample index in ipp
il0_rgs_min = tx_start_samp + 1
il0_rgs_max = rx_end_samp - tx_pulse_samps
il0_min_range_gate = cfg.min_range_gate + il0_rgs_min
il0_max_range_gate = cfg.max_range_gate + il0_rgs_min

rx_inds = np.arange(il0_rg0, il0_rg1)
tx_inds = np.arange(tx_start_samp, tx_end_samp)
# rx = signal[il0_min_range_gate:il0_max_range_gate]
rx = signal[il0_rg0:il0_rg1]
tx = signal[tx_start_samp:tx_end_samp]

# for some reason these are 1 off??
print(il0_rg0, il0_rg1, il0_min_range_gate, il0_max_range_gate)

doppler = v_true * (exp_params.radar_frequency * 1e6) / constants.c
ture_dop_phasor = np.exp(-2j * np.pi * doppler * rx_inds / exp_params.sample_rate)
sub_resolution = 10
sub_resolution_doppler = 50

base_range_gates = np.arange(cfg.min_range_gate, cfg.max_range_gate - len(tx), 1) + 1
base_r_tests = base_range_gates * constants.c / exp_params.sample_rate

range_gates = np.arange(cfg.min_range_gate, cfg.max_range_gate - len(tx), 1.0 / sub_resolution) + 1
r_tests = range_gates * constants.c / exp_params.sample_rate

sample = np.arange(tx.size)


# determine pulse code here

matches = match_pulse_code(
    tx_signal=tx,
    codes=exp_params.code,
    baud_length_usec=exp_params.baud_length_usec,
    t_samp_usec=exp_params.t_samp_usec,
    ipp_t_usec=exp_params.ipp_samps * exp_params.t_samp_usec,
)
start_code_num = np.argmax(matches)

# fig, ax = plt.subplots()
# ax.plot(matches)
# plt.show()

print("guessed code id: ", start_ipp_num % reader.exp_def.code.shape[0])
print("estimated code id: ", start_code_num)

# TODO: this now works with both! implement that we can choose which to do, modulated or simulated
modulated_tx_sim = tx_signal_model(
    code=exp_params.code[start_code_num, :],
    baud_length_usec=exp_params.baud_length_usec,
    t_samp_usec=exp_params.t_samp_usec,
    tx_start_samp=0,
    ipp_samps=exp_params.ipp_samps,
    read_length=len(tx),
    sub_resolution=sub_resolution,
    bandwidth=1e6,
)

# TODO: we can probably use the tx signal to measure phase drift and apply that to the analytic
# model? something like this


def moving_average(x, w):
    x_padded = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(x_padded, np.ones(w), "valid") / w


tx_phase_direct = np.unwrap(2 * np.angle(tx)) / 2
tx_phase_smooth = moving_average(
    tx_phase_direct,
    5 * int(exp_params.baud_length_usec * 1e-6 * exp_params.sample_rate),
)
tx_phase_mod = np.exp(-1j * tx_phase_smooth)
modulated_tx = modulated_tx_sim * tx_phase_mod[:, None]
# modulated_tx = modulated_tx_sim

fig, ax = plt.subplots()
ax.plot(tx_phase_direct)
ax.plot(tx_phase_smooth, ls="--")

modulated_tx_meas = tx_modulation_model(
    tx_signal=tx,
    tx_stencil=np.full(tx.shape, True, dtype=np.bool),
    sub_resolution=sub_resolution,
)
# modulated_tx = modulated_tx_meas

samp_r_true = exp_params.sample_rate * r_true / constants.c + tx_start_samp
fft_len = 2 ** (int(np.log2(len(tx))) + 2)
fvec = fftshift(fftfreq(fft_len, d=1.0 / exp_params.sample_rate))
d_freq = fvec[1] - fvec[0]

samp_inds = np.arange(len(rx))
corrs = np.zeros_like(r_tests)
dopps = np.zeros_like(r_tests)
pbar = tqdm(total=corrs.size)
for ri, r in enumerate(base_r_tests):
    z = rx[ri : (ri + len(tx))]
    for rii in range(sub_resolution):
        ind = ri * sub_resolution + rii
        decoded = z * np.conj(modulated_tx[:, rii])

        spec = fftshift(fft(decoded, n=fft_len))
        mi = np.argmax(np.abs(spec))

        # try:
        #     f_est, phi_est = dft_taylor(
        #         spectrum=spec,
        #         signal_len=len(decoded),
        #         sample_rate=exp_params.sample_rate,
        #     )
        # except RuntimeError:
        #     f_est, phi_est = np.nan, np.nan
        # if np.isnan(f_est):
        # NOTE: apparently this is just better! although its a bit slower
        pwr, f_est, phi_est = dtft_solve(
            decoded_signal=decoded,
            sample_rate=exp_params.sample_rate,
            freq_bracket=(
                fvec[mi] - d_freq,
                fvec[mi] + d_freq,
            ),
        )

        corrs[ind] = pwr
        # corrs[ind] = np.abs(spec[mi])
        dopps[ind] = f_est

        pbar.update(1)
        # if np.abs(r_tests[ind] - r_true) < 300.0:
        #     fig, axes = plt.subplots(3, 1)
        #     axes[0].plot(np.real(decoded))
        #     axes[0].plot(np.imag(decoded))
        #     axes[1].plot(samp_inds, np.real(rx))
        #     axes[1].plot(samp_inds, np.imag(rx))
        #     axes[1].axvline(ri, c="g", ls="--")
        #     axes[1].axvline(ri + len(tx), c="g", ls="--")
        #     axes[2].plot(r_tests, corrs)
        #     axes[2].plot(r_tests[ind], corrs[ind], "or")
        #     plt.show()
pbar.close()

ri_max = np.argmax(corrs)
ri = int(ri_max // sub_resolution)

rg_true = r_true / constants.c * exp_params.sample_rate

rii = int(ri_max - ri * sub_resolution)
z = rx[ri : (ri + len(tx))]
z_dop = z * np.conj(modulated_tx[:, rii])

z = rx[ri : (ri + len(tx))]
dop_phasor = np.exp(-2j * np.pi * dopps[ri_max] * np.arange(len(tx)) / exp_params.sample_rate)
tx_sel = modulated_tx[:, rii]

z_comp = z * dop_phasor
decode = z * dop_phasor * np.conj(modulated_tx[:, rii])
z_dop = z * np.conj(modulated_tx[:, rii])

r_est = r_tests[ri_max]
v_est = dopps[ri_max] / (exp_params.radar_frequency * 1e6) * constants.c

spec = fftshift(fft(z_dop, n=fft_len))
fvec = fftshift(fftfreq(fft_len, d=1 / exp_params.sample_rate))

print(f"{r_est - r_true=} m")
print(f"{v_est - v_true=} m/s")

fig, axes = plt.subplots(3, 1)
axes[0].plot(range_gates, corrs, "-")
axes[0].plot(range_gates[ri_max], corrs[ri_max], "or")
axes[0].axvline(rg_true, c="r")
axes[1].plot(range_gates, corrs, "-x")
axes[1].plot(range_gates[ri_max], corrs[ri_max], "or")
axes[1].axvline(rg_true, c="r")
axes[1].set_xlim(rg_true - 10, rg_true + 10)
axes[2].plot(fvec, np.abs(spec))
# axes[2].set_xlim()

fig, axes = plt.subplots(2, 1)
for ind in range(sub_resolution):
    axes[0].plot(np.real(modulated_tx[:, ind]), label=f"{ind}")
    axes[0].plot(np.imag(modulated_tx[:, ind]))
    axes[1].plot(np.real(modulated_tx_meas[:, ind]))
    axes[1].plot(np.imag(modulated_tx_meas[:, ind]))
axes[0].legend()

fig, axes = plt.subplots(4, 1)
axes[0].plot(np.real(tx_sel / np.max(np.abs(tx_sel))), ls="--", c="b")
axes[0].plot(np.imag(tx_sel / np.max(np.abs(tx_sel))), ls="--", c="r")
axes[0].plot(np.real(z / np.max(np.abs(z))), c="b")
axes[0].plot(np.imag(z / np.max(np.abs(z))), c="r")
axes[1].plot(np.real(tx_sel / np.max(np.abs(tx_sel))), ls="--", c="b")
axes[1].plot(np.imag(tx_sel / np.max(np.abs(tx_sel))), ls="--", c="r")
axes[1].plot(np.real(z_comp / np.max(np.abs(z_comp))), c="b")
axes[1].plot(np.imag(z_comp / np.max(np.abs(z_comp))), c="r")
axes[2].plot(np.real(z_dop / np.max(np.abs(z_dop))), c="b")
axes[2].plot(np.imag(z_dop / np.max(np.abs(z_dop))), c="r")
axes[3].plot(np.real(decode / np.max(np.abs(decode))), c="b")
axes[3].plot(np.imag(decode / np.max(np.abs(decode))), c="r")


fig, ax = plt.subplots()
ax, handles = plotting.rti(
    ax,
    reader,
    start_time=start_time_str,
    end_time=end_time_str,
    start_range_gate=cfg.min_range_gate,
    end_range_gate=cfg.max_range_gate,
    range_gate_unit="sample",
    axis_units=True,
    log=True,
)
# plt.show()


fig, ax = plt.subplots()
ax.plot(np.real(tx))
ax.plot(np.imag(tx))


fig, axes = plt.subplots(2, 1)
axes[0].plot(np.real(data_ipp_vec[:, ipp_index]))
axes[0].plot(np.imag(data_ipp_vec[:, ipp_index]))
axes[0].axvline(samp_r_true - il0_rg0, color="r")

axes[1].plot(np.real(data_ipp_vec_full[:, ipp_index]))
axes[1].plot(np.imag(data_ipp_vec_full[:, ipp_index]))
axes[1].axvline(samp_r_true, color="r")
axes[1].axvline(t_tx_start_samp, color="g", ls="--")
axes[1].axvline(t_tx_end_samp, color="g", ls="--")

plt.show()
