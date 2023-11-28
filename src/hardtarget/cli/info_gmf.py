import h5py
import re
from pathlib import Path
from .commands import add_command


################################################################
# PARSE GMF OUT (H5)
#
# Parse groups and datasets recursively
################################################################


def is_scale(obj):
    return obj.attrs.get("CLASS") == b"DIMENSION_SCALE"


def all_h5_files(gmf_folder):
    """generate all files matching 'yyyy-mm-ddThh-00-00/gmf-*.h5'"""
    top = Path(gmf_folder)
    dir_pattern = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}-00-00')
    subdirs = [d for d in top.iterdir() if d.is_dir() and dir_pattern.match(d.name)]
    file_pattern = re.compile(r'^gmf-.*\.h5$')
    files = []
    for subdir in subdirs: 
        files += [f for f in subdir.iterdir() if f.is_file and file_pattern.match(f.name)]
    return files


def inspect_h5_node(obj, path=[]):
    items = []
    for child_key, child_item in obj.items():
        child_path = path + [child_key]
        if isinstance(child_item, h5py.Group):
            items += inspect_h5_node(child_item, child_path)
        else:
            items.append(inspect_h5_leaf(child_item, child_path))
    return items


def inspect_h5_leaf(obj, path):
    item = {
        "obj": obj
    }
    if isinstance(obj, h5py.Dataset):
        item["type"] = "scale" if is_scale(obj) else "dataset"
    else:
        item["type"] = "other"
    return path, item


################################################################
# MAIN
################################################################


def main(args):

    # find all h5 files in
    files = all_h5_files(args.path)
    # choose one f5 file from withih
    if len(files) == 0:
        print("Found no files at path", args.path)
        return

    file = files[0]
    with h5py.File(file, "r") as f:
        items = inspect_h5_node(f)

        # print

        def path_to_str(path):
            return "".join([f"[{k}]" for k in path])

        for path, item in items:
            _path = path_to_str(path)
            if item['type'] in ["dataset", "scale"]:
                print(f"-- {_path} type:{item['type']} shape:{item['obj'].shape} dtype:{item['obj'].dtype}")


################################################################
# CLI INTEGRATION
################################################################


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with GMF data")
    return parser


add_command(
    name="info_gmf",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script for printing gmf metadata.",
    ),
)
