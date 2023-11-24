import h5py
import re
from pathlib import Path
from .commands import add_command


def parser_build(parser):
    parser.add_argument("path", help="Path to source directory with GMF data")
    return parser


def all_files(top):
    """generate all files matching 'yyyy-mm-ddThh-00-00/gmf-*.h5'"""
    top = Path(top)
    dir_pattern = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}-00-00')
    subdirs = [d for d in top.iterdir() if d.is_dir() and dir_pattern.match(d.name)]
    file_pattern = re.compile(r'^gmf-.*\.h5$')
    files = []
    for subdir in subdirs: 
        files += [f for f in subdir.iterdir() if f.is_file and file_pattern.match(f.name)]
    return files


def is_scale(obj):
    return obj.attrs.get("CLASS") == b"DIMENSION_SCALE"


def path_to_str(path):
    return "".join([f"[{k}]" for k in path])


def inspect_dataset(path, obj):
    _path = path_to_str(path)
    if (is_scale(obj)):
        print(f"-- {_path} (scale) shape {obj.shape}, {obj.dtype}")
    else:
        print(f"-- {_path} (dataset) shape {obj.shape}, {obj.dtype}")


def inspect_other(path, obj):
    _path = ".".join(path)
    print(f"-- [{_path}] shape {obj.shape}, {obj.dtype}")


def inspect_node(path, obj):
    if path is None:
        path = []
    for key, item in obj.items():
        new_path = path + [key]
        if isinstance(item, h5py.Group):
            inspect_node(new_path, item)
        elif isinstance(item, h5py.Dataset):
            inspect_dataset(new_path, item)
        else:
            inspect_other(new_path, item)


def main(args):

    files = all_files(args.path)
    # choose one file
    if len(files) == 0:
        print("Found no files at path", args.path)
        return

    file = files[0]
    with h5py.File(file, "r") as f:
        inspect_node(None, f)


add_command(
    name="info_gmf",
    function=main,
    parser_build=parser_build,
    add_parser_args=dict(
        description="Script for printing gmf metadata.",
    ),
)
