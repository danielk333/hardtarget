import json
from pathlib import Path

FILENAME = "metadata.json"


def write_metadata(dstdir, kw):
    path = Path(dstdir) / FILENAME
    with open(path, "w") as f:
        f.write(json.dumps(kw))


def read_metadata(dstdir):
    path = Path(dstdir) / FILENAME
    with open(path, "r") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    kw = {
        "jalla": 1,
        "palla": 2
    }
    write_metadata(".", kw)
    import pprint
    pprint.pprint(read_metadata("."))