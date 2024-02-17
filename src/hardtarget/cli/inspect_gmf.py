import h5py
from .commands import add_command
from hardtarget.analysis.utils import inspect_h5_node, all_gmf_h5_files


def parser_build(parser):
    """Define argparse sub parser."""
    parser.add_argument("path", help="Path to source directory with GMF data")
    return parser

def main(args):
    """Argparse entrypoint."""

    # find all h5 files in
    files = all_gmf_h5_files(args.path)
    # choose one f5 file from withih
    if len(files) == 0:
        print("Found no files at path", args.path)
        return

    # print
    def path_to_str(path):
        return "".join([f"[{k}]" for k in path])

    with h5py.File(files[0], "r") as f:
        items = inspect_h5_node(f)

    for path, item in items:
        _path = path_to_str(path)
        if item["type"] == "scalar":
            print(f"-- {_path} value:{item['value']} dtype:{item['dtype']}")
        elif item["type"] == "object":
            print(f"-- {_path} value:{item['value']}")
        elif item["type"] == "attributes":
            print(f"-- {_path} attributes: " + "{")
            for key, val in item["attrs"].items():
                print(f"    {key}: {val}")
            print("}")
        elif item["type"] == "dataset":
            if item["scale"]:
                print(f"-- {_path} (scale) shape:{item['shape']} dtype:{item['dtype']}")
            else:
                print(f"-- {_path} (dataset) shape:{item['shape']} dtype:{item['dtype']}")


