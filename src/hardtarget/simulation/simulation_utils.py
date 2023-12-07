from hardtarget.experiments import EXP_FILES

def load_expconfig(xpname):
    cfg_name = xpname + ".ini"
    assert cfg_name in EXP_FILES, "experiment not found in pre-defined configurations"
    cfg_file = EXP_FILES[cfg_name]
    try:
        cfg = configparser.ConfigParser()
        cfg.read_file(open(cfg_file, "r"))
        return cfg
    except Exception as e:
        raise ValueError(f"Couldn't open config file for {xpname}:" + str(e))


DEFAULT_SETTINGS = {
    "r0": 1000e3,
    "v0": 2e3,
    "a0": 80.0,
    "ipp": 10000,
    "tx_len": 2000,
    "bit_len": 100,
    "n_ipp": 100,
    "freq": 230e6,
    "sr": 1000000,
    "snr": 30,
}

