import configparser

from .analysis import analyze_params


####################################################################
# LOAD GMF PARAMS FROM CONFIG
####################################################################

def unpack_config(config):
    d = {}
    for section in analyze_params.CONFIG_SECTIONS:
        if section not in config:
            continue

        for key in config[section].keys():
            # Convert values to specific types
            if key in analyze_params.INT_PARAM_KEYS:
                d[key] = config.getint(section, key)
            elif key in analyze_params.BOOL_PARAM_KEYS:
                d[key] = config.getboolean(section, key)
            elif key in analyze_params.FLOAT_PARAM_KEYS:
                d[key] = config.getfloat(section, key)
            else:
                # string
                d[key] = config.get(section, key).strip("'").strip('"')
    return d


def load_gmf_params(config_file):
    """
    Load a gmf config file into to a dictionary
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return unpack_config(config)
