from pathlib import Path
import configparser

"""
Simple script for writing and reading basic python types to metadata file using configparser
"""


FILENAME = "metadata.ini"


def write_drf_metadata(dstdir, meta):

    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Loop through the dictionary and add data to the ConfigParser object
    for key, value in meta.items():
        _section = "None"
        if type(value) is int:
            _section = "Ints"
        elif type(value) is float:
            _section = "Floats"
        elif type(value) is bool:
            _section = "Bools"
        elif type(value) is str:
            _section = "Strings"
        if not config.has_section(_section):
            config.add_section(_section)
        config[_section][key] = str(value) 

    # Write the data to an INI file
    path = Path(dstdir) / FILENAME
    with open(path, 'w') as f:
        config.write(f)


def read_drf_metadata(dstdir):

    path = Path(dstdir) / FILENAME
    config = configparser.ConfigParser()
    config.read(path)
    meta = {}
    for section in config.sections():
        meta[section] = d = {}
        for key in config[section].keys():
            # Convert values to specific types
            if section == "Ints":
                d[key] = config.getint(section, key)
            elif section == "Bools":
                d[key] = config.getboolean(section, key)
            elif section == "Floats":
                d[key] = config.getfloat(section, key)
            elif section == "Strings":
                d[key] = config.get(section, key).strip("'").strip('"')
            else:
                d[key] = config.get(section, key)
    return meta


if __name__ == "__main__":

    import pprint

    # Sample dictionary with key-value pairs
    meta = {
        'key1': 1,
        'key2': 1.2,
        'key3': True,
        'key4': 'jalla'
    }

    write_drf_metadata(".", meta)

    _meta = read_drf_metadata(".")
    pprint.pprint(meta)
    pprint.pprint(_meta)