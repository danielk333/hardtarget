#!/usr/bin/env python
import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path
from pprint import pprint

import numpy as np
from tqdm import tqdm

import hardtarget
from hardtarget.types.constants import Impl, TargetEstimationMethod

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "sim_test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "sim_test.ini"

try:
    from tabulate import tabulate
except ImportError:
    raise Exception("`pip install tabulate` for nice printing of results")

DEFAULT_IMPLS = [Impl.numpy, Impl.c]

# if len(hardtarget.matched_filter.ANALYSIS_LIBS[Impl.cuda]) > 0:
#    DEFAULT_IMPLS.append(Impl.cuda)

print(f"{DEFAULT_IMPLS=}")


def compute_mf_wrapper(x):
    job, kw = x
    hardtarget.target_estimation(job=job, **kw)


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

    tmp_dir = tempfile.TemporaryDirectory()
    sim_path = Path(tmp_dir.name) / "data_drf"
    analysis_path = Path(tmp_dir.name) / "analysis"

    _, _, _, _, _, _, _, exp = utils.sim_data(
        sim_path,
        config=config,
        range0=range0,
        vel0=vel0,
        acel0=acel0,
    )

    gmfimpl, gmfmethod = gmf_conf

    # process
    t0 = time.time()
    if cores > 1:
        with multiprocessing.Pool(processes=cores) as pool:
            pool.map(
                func=compute_mf_wrapper,
                iterable=[
                    (
                        {"idx": ind, "N": cores},
                        dict(
                            path=sim_path,
                            config=config,
                            method=gmfmethod,
                            implementation=gmfimpl,
                            clobber=True,
                            output=analysis_path,
                            progress=False,
                        ),
                    )
                    for ind in range(cores)
                ],
            )
    else:
        hardtarget.analyse(
            path=sim_path,
            config=config,
            method=gmfmethod,
            implementation=gmfimpl,
            clobber=True,
            output=analysis_path,
            progress=False,
        )
    dt = time.time() - t0
    return dt


def print_results(config, data, headers, title):
    print("\n" + "-" * 10 + f"RESULTS {title}" + "-" * 10)
    print("## CONFIG:")
    pprint(config, indent=4)

    print("\n## DATA:\n")
    print(tabulate(data, headers=headers))
    print("\n")
    print(json.dumps(data, indent=4))
    print("\n")
    print(tabulate(data, headers=headers, tablefmt="latex"))


def fgmf_small_test(total_time, max_rg, cores=1, impls=DEFAULT_IMPLS):
    config = dict(
        range_gate_lims=(6640, max_rg),
        n_ipp=10,
        tau_ipp=5,
        total_time=total_time,
        cores=cores,
    )
    impl_data = []
    for impl in tqdm(impls, desc="impl"):
        dt = run_hardtarget(gmf_conf=(impl, TargetEstimationMethod.fgmf), **config)
        impl_data.append(dt)
    impl_data = [[impl, dt, dt / max(impl_data)] for dt, impl in zip(impl_data, impls)]
    print_results(
        config,
        impl_data,
        headers=["Implementation", "Time [s]", "Time [%]"],
        title="FastGMF implementation",
    )


def fgmf_large_test(total_time, max_rgs, impls=DEFAULT_IMPLS):
    config = dict(
        n_ipp=10,
        tau_ipp=5,
        total_time=total_time,
        cores=1,
    )
    min_rg = 6600
    impl_data = []
    pbar = tqdm(total=len(max_rgs) * 3, desc="impl-rgs sampling")
    for max_rg in max_rgs:
        rg_data = []
        for impl in impls:
            dt = run_hardtarget(
                gmf_conf=(impl, TargetEstimationMethod.fgmf), range_gate_lims=(min_rg, max_rg), **config
            )
            pbar.update(1)
            rg_data.append(dt)
        rg_data = [max_rg - min_rg] + rg_data + [x / max(rg_data) * 100 for x in rg_data]
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


def fgmf_cores_test(total_times, max_rgs, impls=DEFAULT_IMPLS):
    for total_time in total_times:
        config = dict(
            n_ipp=10,
            tau_ipp=5,
            total_time=total_time,
        )
        test_cores = [1, 6]
        min_rg = 6600
        impl_data = []
        pbar = tqdm(total=len(max_rgs) * 3 * len(test_cores), desc="impl-rgs sampling")
        for max_rg in max_rgs:
            for cores in test_cores:
                rg_data = []
                for impl in impls:
                    if cores > 1 and impl == Impl.cuda:
                        rg_data.append(0)
                        continue
                    dt = run_hardtarget(
                        gmf_conf=(impl, TargetEstimationMethod.fgmf),
                        range_gate_lims=(min_rg, max_rg),
                        cores=cores,
                        **config,
                    )
                    pbar.update(1)
                    rg_data.append(dt)
                rg_data = [max_rg - min_rg, cores] + rg_data + [x / max(rg_data) * 100 for x in rg_data]
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


def fgmf_vs_fdpt(cores=1, total_time=10.0, max_rg=6700):
    config = dict(
        range_gate_lims=(6600, max_rg),
        n_ipp=10,
        tau_ipp=5,
        total_time=total_time,
        cores=cores,
    )
    impl_data = []
    impls = [Impl.numpy, Impl.c]
    algs = [TargetEstimationMethod.fgmf, TargetEstimationMethod.fdpt]
    pbar = tqdm(total=len(impls) * len(algs), desc="impls and algs")
    for impl in impls:
        algs_dt = []
        for alg in algs:
            dt = run_hardtarget(gmf_conf=(impl, alg), **config)
            pbar.update(1)
            algs_dt.append(dt)
        impl_data.append(
            [
                impl,
            ]
            + algs_dt
            + [x / max(algs_dt) for x in algs_dt]
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
    "alg_vs": lambda: fgmf_vs_fdpt(cores=1, total_time=10.0, max_rg=6700),
    "alg_vs_long": lambda: fgmf_vs_fdpt(cores=1, total_time=100.0, max_rg=6700),
    "alg_vs_much": lambda: fgmf_vs_fdpt(cores=1, total_time=10.0, max_rg=7700),
    "fgmf-impl": lambda: fgmf_small_test(total_time=4.0, max_rg=6700),
    "fgmf-impl-long": lambda: fgmf_small_test(total_time=20.0, max_rg=8000),
    "fgmf-impl-mat": lambda: fgmf_large_test(total_time=4.0, max_rgs=[6700, 6800, 7000]),
    "fgmf-impl-mat-long": lambda: fgmf_large_test(total_time=20.0, max_rgs=[6700, 7000]),
    "fgmf-impl-cores": lambda: fgmf_cores_test(total_times=[20.0], max_rgs=[6700, 8000]),
    "fgmf-impl-cores-time": lambda: fgmf_cores_test(total_times=[20.0, 200.0], max_rgs=[6700]),
}

parser = argparse.ArgumentParser()
parser.add_argument("--list", action="store_true")
parser.add_argument("scenarios", nargs="+")
args = parser.parse_args()

if args.list:
    print("Scenarios:")
    for key in scenarios:
        print(key)

for s in args.scenarios:
    if s not in scenarios:
        raise ValueError(f"{s} not in scenarios")
    func = scenarios[s]
    func()
