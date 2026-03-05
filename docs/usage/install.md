# Install
---

!!! note
    Hardtarget installation is only supported on Linux

---


## Prerequisites
Hardtarget depends on `gcc`, `libhdf5-dev`, and `libfftw3-dev`.


```bash
    sudo apt install gcc libfftw3-dev libhdf5-dev
```

It is recommended to install Hardtarget in a virtual environment, e.g. `virtualenv` or `conda`.

```bash
    $ python3.X -m venv venv
    $ source venv/bin/activate
    $ pip install --upgrade pip
```

---

## Installation from PyPi

```bash

    $ pip install hardtarget
```

---

## Installation from Git
Basic install.

```bash

    $ git clone git@github.com:danielk333/hardtarget.git
    $ cd hardtarget
    $ pip install .[plotting]
```

Nightly build.

```bash

    $ git clone --branch develop git@github.com:danielk333/hardtarget.git
    $ cd hardtarget
    $ pip install .
```

Full developer install.

```bash

    $ pip install -e .[mpi,develop,plotting,profiling]
```