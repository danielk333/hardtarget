# CLI

The Command Line Interface of Hardtarget is exposed under a single executable **hardtarget**.

```bash
    hardtarget -h
```

The **hardtarget** executable provides access to 4 tools, each with subcommands.
The option **-v** turns on verbose mode.

---

## Inspect

Use the **inspect** tool to inspect a data product either the raw data or analysed data.
The tool prints key properties and metadata to the screen. Currently, **inspect**
functionality is supported for any data supported by [radardef](https://pypi.org/project/radardef/)

```bash
    $ hardtarget inspect -h
    $ hardtarget inspect drf -h
    $ hardtarget inspect gmf -h
    $ hardtarget inspect drf /data/drf
    $ hardtarget inspect gmf /data/gmf
```

---

## Plot

Use the **plot** tool to plot a data product of given type. Currently, **plot**
functionality is supported for **raw data** (supported by radardef) and all types of analyses.

```bash
    $ hardtarget plot -h
    $ hardtarget plot raw -h
    $ hardtarget plot analysis -h
    $ hardtarget plot raw /data/raw -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30
    $ hardtarget plot analysis <method> /data/mf --config path/to/cfg -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30
```

!!! note
Plot only a small time range. Use the (**inspect**)[#inspect] tool to figure out time ranges.

---

## Check

Use the **check** tool to perform certain checks. Currently, checks are included for
cuda support and range gate configuration.

```bash
     $ hardtarget check -h
     $ hardtarget check cuda
     $ hardtarget check range-gates -h
```

---

## Analyse

Use the **analyse** tool to launch the analysis. The different analysis methods are\_

```bash
    $ hardtarget target_estimation /data/path --config cfg.ini --progress -o /data/
    $ hardtarget optimization /data/path --config cfg.ini --progress -o /data/
    $ hardtarget event_detection /data/path --config cfg.ini --progress -o /data/
    $ hardtarget direction_of_arrival /data/path <radar_station> --config cfg.ini --progress -o /data/

```
