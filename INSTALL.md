## Install

These are extended installation instructions.

### Clone repo

```bash
git clone git@github.com:jvierine/hard_target.git
```

Make sure you are on the right branch. It should probably be _main_ or _develop_.

Check active branch
```bash
git branch
```

Switch to branch develop
```bash
git checkout develop
```


### Check pre-requisites

The package depends on _gcc_ and _libfftw3-dev_.
These may for instance be installed using apt.

```bash
sudo apt install gcc libfftw3-dev
```

In addition, if cuda acceleration is needed, this requires nvidia GPU hardware and a working cuda environment.

```bash
sudo apt install nvidia-cuda-toolkit
```

You may test you setup using with a script

```bash
python3 src/gmf_cuda_lib/testcuda.py
```

### Create virtual environment

Create environment in folder _venv_.

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









