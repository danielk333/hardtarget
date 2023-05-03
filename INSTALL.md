## Install

These are extended installation instructions.

- [Check prerequisites](#prerequisites)
- [Make virtual environment](#virtual-environment-cheatsheet)
- [Install PyPi](#install-pypi)
- [Install Local Repository](#install-local-repository)


### Prerequisites

The package depends on _gcc_ and _libfftw3-dev_.
These may for instance be installed using apt.

```bash
sudo apt install gcc libfftw3-dev
```

In addition, if cuda acceleration is needed, this requires nvidia GPU hardware and a working cuda environment.

```bash
sudo apt install nvidia-cuda-toolkit
```


### Virtual Environment Cheatsheet

Create environment in folder _venv_. This requires _python3.*-venv_ to be installed on the system.

```bash
cd hard_target
python3 -m venv venv
```

Activate the environment

```bash
source venv/bin/activate
```

Initialise the environment

```bash
pip install --upgrade pip 
```

Deactivate the environment

```bash
deactivate
```

Delete the environment

```bash
rm -rf venv
```

### Install PyPi

Hardtarget can be installed from _PyPi_. (CURRENTLY NOT WORKING)

```bash
pip install hardtarget
```

### Install Local Repository

Hardtarget can be installed from source code in git repository.

Clone repository
```bash
git clone git@github.com:jvierine/hard_target.git
cd hard-target
```

Make sure you are on the intended branch. It should probably be _main_ or _develop_.

Check active branch
```bash
git branch
```

Switch to other branch if needed
```bash
git checkout develop
```

Install from repository
```bash
pip install .
```









