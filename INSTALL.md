## Install

These are extended installation instructions.

- [Check prerequisites](#prerequisites)
- [Make virtual environment](#virtual-environment-cheatsheet)
- [Install from PyPi](#install-from-pypi)
- [Install from Git](#install-from-git)


### Prerequisites

### System Packages

The package depends on _gcc_ and _libfftw3-dev_.
```bash
sudo apt install gcc libfftw3-dev
```

#### Python Version

Python >= 3.7 installed on system.

For example, install python3.8.
```bash
sudo apt install python3.8
sudo apt install python3.8-venv
```

#### MPI

Optional MPI support.

Make sure the Python dev package for the specific Python version is installed. For instance:

```bash
sudo apt-get install python3.8-dev
```

Then install MPI package

```bash
sudo apt install python3-mpi4py
```

#### Cuda

Optional Cuda support.

Cuda support may be tricky to set up, and a system install may also
be unwanted. For these reasons it may be more convenient to use 
a container environment such as Docker to unlock Cuda support.

#### Docker

TODO


### Virtual Environment Cheatsheet

Create environment in folder _venv_. 

```bash
python3 -m venv venv
```

Or specify Python version.
```bash
python3.8 -m venv venv
```

Activate the environment.
```bash
source venv/bin/activate
```

Initialise the environment.
```bash
pip install --upgrade pip 
```

Deactivate the environment.
```bash
deactivate
```

Delete the environment.
```bash
rm -rf venv
```

### Install from PyPi

Hardtarget can be installed from _PyPi_. 

(CURRENTLY NOT WORKING!)

```bash
pip install hardtarget
```

### Install from Local Repository

Hardtarget can be installed from source code in git repository.

Clone repository.
```bash
git clone git@github.com:jvierine/hard_target.git
cd hard-target
```

Make sure you are on the intended branch. It should probably be _main_ or _develop_.

Check active branch.
```bash
git branch
```

Switch to other branch if needed.
```bash
git checkout develop
```

Install from local repository
```bash
pip install .
```

#### Developer Install

Install with additional developer dependencies

```bash
pip install .[develop,mpi,plotting]
```

It is also possible to install in developer mode, to avoid repeated reinstalling of the package after code modifications.

```bash
pip install -e .
```
