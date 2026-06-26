import re
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from hardtarget.constants import MethodAbbreviation


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


def get_analysed_h5_files(path: str | Path, method: Optional[MethodAbbreviation] = None) -> list[Path]:
    """
    Collects paths to each file in the directory matching the analysed output naming convention

    Args:
        path: Directory containing the analyse output.
        method: If data from a specific method should be retrived.

    Returns:
        List of files matching the output naming convention.
    """

    if not method:
        file_pattern = re.compile(rf"(^|,)({'|'.join(abb for abb in MethodAbbreviation)})-.*\.h5$")
    else:
        file_pattern = re.compile(rf"^{method}-.*\.h5")

    files = []
    for _path in [p for p in Path(path).rglob("*.*")]:
        if re.match(file_pattern, _path.name):
            files.append(_path)

    return files
