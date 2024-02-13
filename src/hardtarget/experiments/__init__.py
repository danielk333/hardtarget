import pathlib
import importlib.resources
import configparser
import numpy as np

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


def load_radar_code(xpname):
    code_name = xpname + "_code.txt"
    assert code_name in EXP_FILES, "radar code not found in pre-defined configurations"
    code_file = EXP_FILES[code_name]
    try:
        with open(code_file, "r") as fh:
            code = []
            for line in fh:
                code.append([1 if ch == '+' else -1 for ch in line.strip()])
        code = np.array(code)
        return code
    except Exception as e:
        raise ValueError(f"Couldn't open code file for {xpname}:" + str(e))


def load_expconfig(xpname):
    cfg_name = xpname + ".ini"
    assert cfg_name in EXP_FILES, f'experiment "{cfg_name}" not found in pre-defined configurations'
    cfg_file = EXP_FILES[cfg_name]
    try:
        cfg = configparser.ConfigParser()
        cfg.read_file(open(cfg_file, "r"))
        return cfg
    except Exception as e:
        raise ValueError(f"Couldn't open config file for {xpname}:" + str(e))