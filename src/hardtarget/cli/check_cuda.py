from .commands import add_command


def parser_build(parser):
    return parser


def main(args):
    try:
        import hardtarget.gmf.gmf_cuda as gcu
        gcu.print_cuda_devices()
    except ImportError as e:
        print(e)


if __name__ == '__main__':
    main()
