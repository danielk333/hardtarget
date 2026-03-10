# Quickstart

These are the basic steps working with radar data using Hardtarget. This is the CLI approach, more detailed
examples for each method can be found at [target estimation example](../examples/target_estimation/analyse_and_inspect.py), [optimization example](../examples/optimization/optimization.py), [event detection example](../examples/event_detection/event_detection.py) and [direction of arrival example](../examples/direction_of_arrival/interferometry.py).

---

## Analyse Data

Run analysis using the following script. The config file `cfg.ini` describes
processing parameters.

```bash

    $ hardtarget <method> /data/path --config cfg.ini --progress -o /data/tmp
```

The script will make the result available in the given folder.

```bash

    $ tree /data/tmp -L 2
    $ data/tmp
    $ └── 2021-04-12T12-00-00
    $     ├── mf-1618229740000000.h5
    $     ├── .......................
    $     └── mf-1618229768000000.h5
```

For large products, analysis may take some time. It is also possible to limit
the conversion to a time range. Either with a timepoint string or milliseconds since epoch (relative to file start if wanted)

```bash

    $ hardtarget <method> /data/path --config cfg.ini --progress -o /data/tmp -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30
```

---

## Plot GMF Data

GMF data may be presented using the Hardtarget plotting tool.

```bash

    $ hardtarget plot gmf /data/gmf -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30
```
