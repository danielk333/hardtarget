"""
The CLI Inspect MF functionality, abstracts the inspect functionality for analysed files to a user friendly
CLI interface.
"""

import argparse

import h5py

from hardtarget.utils.h5_tools import get_analysed_h5_files, inspect_h5_node


def parser_build(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Define argparse sub parser."""
    parser.add_argument("path", help="path to source directory with MF data")
    return parser


def main(args: argparse.Namespace) -> None:
    """Inspect MF CLI"""

    # find all h5 files in
    files = get_analysed_h5_files(args.path)
    # choose one f5 file from withih
    if len(files) == 0:
        print("Found no files at path", args.path)
        return

    # print
    def path_to_str(path: list[str]) -> str:
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
