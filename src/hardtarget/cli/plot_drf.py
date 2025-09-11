import datetime
import numpy as np
from hardtarget import plotting
import matplotlib.pyplot as plt
from hardtarget.drf_utils import load_hardtarget_drf


def parser_build(parser):
    parser.add_argument("path", help="path to source directory with Digital_RF data")
    parser.add_argument("-s", "--start_time", default=None)
    parser.add_argument("-e", "--end_time", default=None)
    parser.add_argument("-c", "--chunk_size", default=None)
    parser.add_argument("--relative_time", action="store_true")
    parser.add_argument("--axis_units", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--keep-tx", action="store_true")
    parser.add_argument("--monostatic", action="store_true")
    parser.add_argument(
        "--start-range",
        default=None,
        help="Desired starting range in given unit, can have negative values",
    )
    parser.add_argument(
        "--end-range",
        default=None,
        help="Desired ending range in given unit",
    )
    parser.add_argument(
        "-u",
        "--unit",
        choices=["sample", "m", "km", "R_E", "LD", "AU"],
        help="Unit for start and end ranges, default [km]",
        default="km",
    )
    return parser


def main(args):
    reader, params = load_hardtarget_drf(args.path)

    if args.relative_time:
        args.start_time = float(args.start_time)
        args.end_time = float(args.end_time)

    if args.start_range is not None:
        args.start_range = float(args.start_range)
    if args.end_range is not None:
        args.end_range = float(args.end_range)

    if args.chunk_size is not None:
        args.chunk_size = float(args.chunk_size)
        chnl = params["rx_channel"]
        props = reader.get_properties(chnl)
        bounds = list(reader.get_bounds(chnl))
        sample_rate = props["samples_per_second"].astype(np.int64)
        dt0 = datetime.datetime.utcfromtimestamp(bounds[0] / sample_rate)
        dt1 = datetime.datetime.utcfromtimestamp(bounds[1] / sample_rate)

        if args.start_time is None:
            args.start_time = dt0
        elif not args.relative_time:
            args.start_time = np.datetime64(args.start_time).astype("datetime64[us]").astype("int64")*1e-6
            args.start_time = datetime.datetime.utcfromtimestamp(args.start_time)
        if args.end_time is None:
            args.end_time = dt1
        elif not args.relative_time:
            args.end_time = np.datetime64(args.end_time).astype("datetime64[us]").astype("int64")*1e-6
            args.end_time = datetime.datetime.utcfromtimestamp(args.end_time)
        dt = dt1 - dt0
        chunks_total = int(dt.total_seconds() // args.chunk_size)
        chunks = [
            (
                args.start_time + chunk * dt / chunks_total,
                args.start_time + (chunk + 1) * dt / chunks_total,
            )
            for chunk in range(chunks_total - 1)
        ]
        chunks.append((chunks[-1][1], args.end_time))
    else:
        chunks = [(args.start_time, args.end_time)]

    for ind, (t0, t1) in enumerate(chunks):
        print(f"Plotting chunk {ind} [{t0}, {t1}]")
        fig, ax = plt.subplots()
        ax, handles = plotting.rti(
            ax,
            reader,
            params,
            start_time=t0,
            end_time=t1,
            relative_time=args.relative_time,
            axis_units=args.axis_units,
            log=args.log,
            start_range_gate=args.start_range,
            end_range_gate=args.end_range,
            range_gate_unit=args.unit.strip().lower(),
            monostatic=args.monostatic,
            keep_tx=args.keep_tx,
        )
        plt.show()
