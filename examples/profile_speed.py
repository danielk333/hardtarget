#!/usr/bin/env python
from pprint import pprint
from tqdm import tqdm
import time
import numpy as np
import tempfile
import hardtarget
import argparse
import json
import multiprocessing

try:
    from tabulate import tabulate
except ImportError:
    print("`pip install tabulate` for nice printing of results")
    exit(1)


def compute_gmf_wrapper(x):
    job, kw = x
    hardtarget.compute_gmf(job=job, **kw)


def run_hardtarget(
    range_gate_lims,
    n_ipp,
    tau_ipp,
    total_time,
    gmf_conf,
    cores=1,
    range0=2000e3,
    vel0=0.4e3,
    acel0=0.10e3,
):

    frequency_decimation = 10
    experiment_params = {
        "sample_rate": 1000000,
        "ipp": 20000,
        "tx_pulse_length": 1920.0,
        "tx_start": 82.0,
        "tx_end": 2002.0,
        "rx_start": 0,
        "rx_end": 20000,
        "cal_on": 19900.0,
        "cal_off": 19997.0,
        "frequency": 929.6,
        "baud_length": 30.0,
        "code": hardtarget.load_radar_code("leo_bpark"),
    }
    config_str = f"""

    [signal-processing]
        n_ipp={n_ipp}
        ipp_offset=0
        min_range_gate={range_gate_lims[0]}
        max_range_gate={range_gate_lims[1]}
        min_acceleration=-300.0
        max_acceleration=300.0
        range_gate_step=1
        frequency_decimation={frequency_decimation}
        num_cohints_per_file=10
        node_gpus=1
        dpt_ipp_delay_parameter={tau_ipp}

    """

    gmfimpl, gmfmethod = gmf_conf

    t_end = total_time
    coh_int_len = experiment_params["ipp"] * 1e-6 * n_ipp
    t_abs = np.arange(0, t_end + coh_int_len, coh_int_len)

    simulation_params = {
        "epoch": "2021-04-12T12:15:40",
        "start_time": 0,
        "end_time": t_end,
        "target_start_time": 0,
        "target_end_time": t_end,
        "noise_sigma": 0,
        "tx_amp": 1,
    }

    rx_channel = "sim"

    def range_function(t):
        inds = np.logical_and(t >= t_abs[0], t <= t_abs[-1])
        if np.any(inds):
            return range0 + vel0 * t[inds] + acel0 * 0.5 * t[inds] ** 2
        else:
            return np.nan

    with (
        tempfile.TemporaryDirectory() as tmp_sim_path,
        tempfile.TemporaryDirectory() as tmp_analysis_path,
        tempfile.NamedTemporaryFile(mode="w+") as tmp_config,
    ):
        tmp_config.write(config_str)
        tmp_config.seek(0)
        tmp_config_path = tmp_config.name

        hardtarget.simulation.drf(
            tmp_sim_path,
            range_function,
            simulation_params,
            experiment_params,
            chnl=rx_channel,
            snr_function=None,
            dtype=np.complex64,
            clobber=True,
        )

        reader, params = hardtarget.drf_utils.load_hardtarget_drf(tmp_sim_path)
        # process
        t0 = time.time()
        if cores > 1:
            with multiprocessing.Pool(processes=cores) as pool:
                pool.map(
                    func=compute_gmf_wrapper,
                    iterable=[
                        (
                            {"idx": ind, "N": cores},
                            dict(
                                rx=(tmp_sim_path, rx_channel),
                                tx=(tmp_sim_path, rx_channel),
                                config=tmp_config_path,
                                gmf_method=gmfmethod,
                                gmf_implementation=gmfimpl,
                                clobber=True,
                                output=tmp_analysis_path,
                                progress=False,
                                subprogress=False,
                            ),
                        )
                        for ind in range(cores)
                    ],
                )
        else:
            hardtarget.compute_gmf(
                rx=(tmp_sim_path, rx_channel),
                tx=(tmp_sim_path, rx_channel),
                config=tmp_config_path,
                gmf_method=gmfmethod,
                gmf_implementation=gmfimpl,
                clobber=True,
                output=tmp_analysis_path,
                progress=False,
                subprogress=False,
            )
        dt = time.time() - t0
    return dt


def print_results(config, data, headers, title):
    print("\n" + "-"*10 + f"RESULTS {title}" + "-"*10)
    print("## CONFIG:")
    pprint(config, indent=4)

    print("\n## DATA:\n")
    print(tabulate(data, headers=headers))
    print("\n")
    print(json.dumps(data, indent=4))


def fgmf_small_test(total_time, max_rg, cores=1):
    config = dict(
        range_gate_lims = (6640, max_rg),
        n_ipp = 10,
        tau_ipp = 5,
        total_time = total_time,
        cores = cores,
    )
    impl_data = []
    impls = ["numpy", "c", "cuda"]
    for impl in tqdm(impls, desc="impl"):
        dt = run_hardtarget(
            gmf_conf = (impl, "fgmf"),
            **config
        )
        impl_data.append(dt)
    impl_data = [
        [impl, dt, dt/max(impl_data)]
        for dt, impl in zip(impl_data, impls)
    ]
    print_results(
        config,
        impl_data,
        headers=["Implementation", "Time [s]", "Time [%]"],
        title="FastGMF implementation",
    )


def fgmf_large_test(total_time, max_rgs):
    config = dict(
        n_ipp = 10,
        tau_ipp = 5,
        total_time = total_time,
        cores = 1,
    )
    min_rg = 6600
    impl_data = []
    pbar = tqdm(total=len(max_rgs)*3, desc="impl-rgs sampling")
    for max_rg in max_rgs:
        rg_data = []
        for impl in ["numpy", "c", "cuda"]:
            dt = run_hardtarget(
                gmf_conf = (impl, "fgmf"),
                range_gate_lims = (min_rg, max_rg),
                **config
            )
            pbar.update(1)
            rg_data.append(dt)
        rg_data = [max_rg - min_rg] + rg_data + [x/max(rg_data)*100 for x in rg_data]
        impl_data.append(rg_data)
    pbar.close()
    print_results(
        config,
        impl_data,
        headers=[
            "Range gates",
            "numpy [s]",
            "c [s]",
            "cuda [s]",
            "numpy [%]",
            "c [%]",
            "cuda [%]",
        ],
        title="FastGMF implementation vs range gate size",
    )


def fgmf_cores_test(total_time, max_rgs):
    config = dict(
        n_ipp = 10,
        tau_ipp = 5,
        total_time = total_time,
    )
    test_cores = [1, 6]
    min_rg = 6600
    impl_data = []
    pbar = tqdm(total=len(max_rgs)*3*len(test_cores), desc="impl-rgs sampling")
    for max_rg in max_rgs:
        for cores in test_cores:
            rg_data = []
            for impl in ["numpy", "c"]:
                dt = run_hardtarget(
                    gmf_conf = (impl, "fgmf"),
                    range_gate_lims = (min_rg, max_rg),
                    cores = cores,
                    **config
                )
                pbar.update(1)
                rg_data.append(dt)
            dt = run_hardtarget(
                gmf_conf = ("cuda", "fgmf"),
                range_gate_lims = (min_rg, max_rg),
                cores = 1,
                **config
            )
            pbar.update(1)
            rg_data.append(dt)
            rg_data = [max_rg - min_rg, cores] + rg_data + [x/max(rg_data)*100 for x in rg_data]
            impl_data.append(rg_data)
    pbar.close()
    print_results(
        config,
        impl_data,
        headers=[
            "Range gates",
            "Cores",
            "numpy [s]",
            "c [s]",
            "cuda [s]",
            "numpy [%]",
            "c [%]",
            "cuda [%]",
        ],
        title="FastGMF implementation vs range gate size",
    )


def fgmf_vs_fdpt(cores=1):
    config = dict(
        range_gate_lims = (6600, 6700),
        n_ipp = 10,
        tau_ipp = 5,
        total_time = 10.0,
        cores = cores,
    )
    impl_data = []
    impls = ["numpy", "c"]
    algs = ["fgmf", "fdpt"]
    pbar = tqdm(total=len(impls)*len(algs), desc="impls and algs")
    for impl in impls:
        algs_dt = []
        for alg in algs:
            dt = run_hardtarget(
                gmf_conf = (impl, alg),
                **config
            )
            pbar.update(1)
            algs_dt.append(dt)
        impl_data.append(
            [impl,] + algs_dt + [x/max(algs_dt) for x in algs_dt]
        )
    pbar.close()
    print_results(
        config,
        impl_data,
        headers=["Implementation"]
        + [f"Time {alg} [s]" for alg in algs]
        + [f"Time {alg} [%]" for alg in algs],
        title="Implementation & method",
    )


scenarios = {
    "alg_vs": fgmf_vs_fdpt,
    "fgmf-impl": lambda: fgmf_small_test(total_time=4.0, max_rg=6700),
    "fgmf-impl-long": lambda: fgmf_small_test(total_time=100.0, max_rg=8000),
    "fgmf-impl-mat": lambda: fgmf_large_test(total_time=4.0, max_rgs=[6700, 6800, 7000]),
    "fgmf-impl-mat-long": lambda: fgmf_large_test(total_time=20.0, max_rgs=[6700, 7000]),
    "fgmf-impl-cores": lambda: fgmf_cores_test(total_time=20.0, max_rgs=[6700, 7000]),
}

parser = argparse.ArgumentParser()
parser.add_argument("--list", action="store_true")
parser.add_argument("scenarios", nargs="+")
args = parser.parse_args()

if args.list:
    print("Scenarios:")
    for key in scenarios:
        print(key)
    exit(0)

for s in args.scenarios:
    if s not in scenarios:
        raise ValueError(f"{s} not in scenarios")
    func = scenarios[s]
    func()
