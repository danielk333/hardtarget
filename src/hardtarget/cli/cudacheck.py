#!/usr/bin/env python3

from .commands import add_command


def parser_build(parser):
    return parser


def main(args):
    import hardtarget.gmf.gmf_cuda as gcu
    gcu.print_cuda_devices()


add_command(
    name="checkcuda",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script for checking cuda devices.",
    ),
)

if __name__ == '__main__':
    main()
