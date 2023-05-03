## Install

These are extended installation instructions.

- [Check prerequisites](#prerequisites)
- [Make virtual environment](#virtual-environment-cheatsheet)
- [Install from PyPi](#install-from-pypi)
- [Install from Git](#install-from-git)


### Prerequisites

The package depends on _gcc_ and _libfftw3-dev_.
```bash
sudo apt install gcc libfftw3-dev
```

Cuda acceleration requires nvidia GPU hardware and a working cuda environment.
```bash
sudo apt install nvidia-cuda-toolkit
```


### Virtual Environment Cheatsheet

Create environment in folder _venv_. 
```bash
python3 -m venv venv
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

### Install from Git

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

Install from repository.
```bash
pip install .
```
