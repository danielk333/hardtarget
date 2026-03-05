# Quickstart

These are the basic steps working with radar data using Hardtarget. This is the CLI approach, more detailed
example can be found at [analyse and inspect](../examples/analysis/analyse_and_inspect.py).

---

## Analyse Data

Run analysis using the following script. The config file `cfg.ini` describes
processing parameters.

```bash

    $ hardtarget -v analyse /data/path --config cfg.ini --progress -o /data/gmf
```

The script will make the result available in the given folder.

```bash

    $ tree /data/path -L 2
    $ 20210412/gmf
    $ └── 2021-04-12T12-00-00
    $     ├── gmf-1618229740000000.h5
    $     ├── .......................
    $     └── gmf-1618229768000000.h5
```

For large products, analysis may take some time. It is also possible to limit
the conversion to a time range.

```bash

    $ hardtarget -v analyse /data/path --config cfg.ini --progress -o /data/gmf -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30
```


---

## Plot GMF Data


GMF data may be presented using the Hardtarget plotting tool.

```bash

    $ hardtarget plot gmf /data/gmf -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30
```





