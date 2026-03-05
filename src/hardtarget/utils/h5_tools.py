import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def is_scale(obj: h5py.Dataset) -> bool:
    """Dataset is a scalar"""
    return obj.attrs.get("CLASS") == b"DIMENSION_SCALE"


def inspect_h5_node(obj: h5py.Group, path: list[str] = []) -> list[tuple[list[str], dict[str, Any]]]:
    """Gather information of each object in a h5 node"""
    items = []
    for child_key, child_item in obj.items():
        child_path = path + [child_key]
        if isinstance(child_item, h5py.Group):
            items += inspect_h5_node(child_item, child_path)
        else:
            items.append(inspect_h5_leaf(child_item, child_path))
    items.append(inspect_h5_attributes(obj, path))
    return items


def inspect_h5_attributes(obj: h5py.Group, path: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Gather attribute information"""
    item = {
        "type": "attributes",
        "attrs": {key: val for key, val in obj.attrs.items()},
    }
    return path, item


def inspect_h5_leaf(
    obj: h5py.Dataset, path: list[str]
) -> tuple[list[str], dict[str, bool | str | h5py.Dataset | np.dtype]]:
    """h5 leaf information"""

    item: dict[str, bool | str | h5py.Dataset | np.dtype] = {}

    item["scale"] = is_scale(obj)
    item["dtype"] = obj.dtype
    item["value"] = obj[()]

    if obj.dtype == "object":
        item["type"] = "object"
    elif obj.shape == ():
        item["type"] = "scalar"
    else:
        item["type"] = "dataset"
        item["shape"] = obj.shape

    return path, item


def get_analysed_h5_files(mf_folder: str | Path) -> list[Path]:
    """
    Collects paths to each file in the directory matching the analysed output naming convention

    ```
        naming convention: 'yyyy-mm-ddThh-00-00/mf-*.h5'
    ```
    Args:
        mf_folder: Directory containing the analyse output.

    Returns:
        List of files matching the output naming convention.
    """

    top = Path(mf_folder)
    dir_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-00-00")
    subdirs = [d for d in top.iterdir() if d.is_dir() and dir_pattern.match(d.name)]
    file_pattern = re.compile(r"^mf-.*\.h5$")
    files = []
    if len(subdirs) == 0:
        files += [f for f in top.iterdir() if f.is_file() and file_pattern.match(f.name)]
    for subdir in subdirs:
        files += [f for f in subdir.iterdir() if f.is_file() and file_pattern.match(f.name)]
    return files
