import digital_rf as drf
import datetime
import numpy as np
import pprint
from .commands import add_command


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with Digital_RF data")
    return parser


def main(args, cli_logger):
    reader = drf.DigitalRFReader(args.path)

    channels = reader.get_channels()
    for chnl in channels:
        print(f"CHANNEL: {chnl}")
        props = reader.get_properties(chnl)
        print("props:")
        pprint.pprint(props)
        bounds = list(reader.get_bounds(chnl))
        print(f"bounds: {bounds}")
        print(f"Total mega-samples: {(bounds[1] - bounds[0])*1e-6}")
        sample_rate = props["samples_per_second"].astype(np.int64)
        dt0 = datetime.datetime.utcfromtimestamp(bounds[0]/sample_rate)
        print(f"start: {dt0}")
        dt1 = datetime.datetime.utcfromtimestamp(bounds[1]/sample_rate)
        print(f"end: {dt1}")
        blocks = reader.get_continuous_blocks(bounds[0], bounds[1], chnl)
        print(f"blocks: {blocks}")
        data = reader.read_vector(bounds[0], 10, chnl)
        print(f"data: {data}")


add_command(
    name="drfinfo",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script for printing drf metadata.",
    ),
)
