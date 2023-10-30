import datetime
import numpy as np
from collections import OrderedDict
from hardtarget import drf_utils

import pprint

# import pprint
from .commands import add_command


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with Digital_RF data")
    return parser


def main(args, cli_logger):

    drf_reader, drf_params = drf_utils.load_hardtarget_drf(args.path)

    d = {
        "drf": [],
        "exp": OrderedDict(drf_params)
    }
    channels = drf_reader.get_channels()
    for chnl in channels:
        props = drf_reader.get_properties(chnl)
        bounds = list(drf_reader.get_bounds(chnl))
        sample_rate = props["samples_per_second"].astype(np.int64)
        dt0 = datetime.datetime.utcfromtimestamp(bounds[0]/sample_rate)
        dt1 = datetime.datetime.utcfromtimestamp(bounds[1]/sample_rate)
        cont_blocks = drf_reader.get_continuous_blocks(bounds[0], bounds[1], chnl)
        cont_blocks = [(k, v) for k, v in cont_blocks.items()]
        mega_samples = (bounds[1] - bounds[0])*1e-6

        d["drf"].append(OrderedDict(
            channel=chnl,
            start=f"{dt0}",  # dt.strftime("%Y-%m-%dT%H:%M:%S")
            end=f"{dt1}",
            bounds=bounds,
            continuous_blocks=cont_blocks,
            mega_samples=mega_samples,
        ))

    pprint.pprint(d)


add_command(
    name="info_drf",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script for printing drf metadata.",
    ),
)
