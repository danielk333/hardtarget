import pathlib
import importlib.resources

EXP_FILES = {}

# To be compatible with 3.7-8
# as resources.files was introduced in 3.9
if hasattr(importlib.resources, "files"):
    _data_folder = importlib.resources.files("hardtarget.experiments")
    for file in _data_folder.iterdir():
        if not file.is_file():
            continue
        if file.name.endswith(".py"):
            continue

        EXP_FILES[file.name] = file

else:
    _data_folder = importlib.resources.contents("hardtarget.experiments")
    for fname in _data_folder:
        with importlib.resources.path("hardtarget.experiments", fname) as file:
            if not file.is_file():
                continue
            if file.name.endswith(".py"):
                continue

            EXP_FILES[file.name] = pathlib.Path(str(file))
