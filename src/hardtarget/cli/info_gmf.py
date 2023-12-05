import re
from pathlib import Path
from .commands import add_command
from hardtarget.gmf_out_utils import load_gmf_out


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

    # print
    def path_to_str(path):
        return "".join([f"[{k}]" for k in path])

    items = load_gmf_out(files[0])
    for path, item in items:
        _path = path_to_str(path)
        if item["type"] == "scalar":
            print(f"-- {_path} value:{item['value']} dtype:{item['dtype']}")
        elif item["type"] == "string":
            print(f"-- {_path} value:{item['value']}")
        elif item["type"] == "dataset":
            if item["scale"]:
                print(f"-- {_path} (scale) shape:{item['shape']} dtype:{item['dtype']}")
            else:
                print(f"-- {_path} (dataset) shape:{item['shape']} dtype:{item['dtype']}")


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
