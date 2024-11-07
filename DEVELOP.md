TODO: move all these into docs folder

### Compiling CUDA code

Compiling cuda code can be fairly tricky as its hard to predict all combinations of GPUs with
different compute capability ( see [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) )
and host compilers (such as clang, gcc, ...). As such, here are some general steps to follow
if the default python build system fails to build the CUDA extensions and one has to build
them manually using the `Makefile` in the source folder.
 
If the default g++ compiler is not compatible with nvcc but you have one installed that is
(see [cuda-installation-guide-linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id11) )
then use the `-ccbin` flag to point to a comaptible version, often the system already has
found it and can be found in the env variable `$NVCC_CCBIN`, and one can do
`make HOST_COMPILER="$NVCC_CCBIN"`.

However, this is not always enough, often one has to specify the computational architecture
of the GPU for the compilation to succeed, also called the Compute Capability of the card.
This can be found by running `nvidia-smi --query-gpu=compute_cap --format=csv`. 
Then this capability can be targeted by `nvcc` when compiling the code by adding
`-gencode arch=compute_75,code=sm_75` to the command.

Or one can use the `deviceQuery` utility that is included in the cuda installation.

The easiest way to work with compiling the extension manually is to clone this repository, 
run `pip install -e .`  and then run `make` in the root. This will go into both the C and
CUDA extensions, run make on them individually and link the binaries into the correct place 
in the source code. Then one does not have to re-install the package to develop and 
modify the compiled code when testing it using python.

### Generating build commands

To generate the `compile_commands.json` needed by the `clang` toolchain use
Build EAR, or [Bear](https://github.com/rizsotto/Bear), and the supplied
Makefiles by running

```bash
bear -- make 
```

in the source directories. Then to combine the json files we can use `jq`. The
complete script would be (assuming we start in repository root)

```bash
cd src/gmf_c_lib
bear -- make
cd ../src/gmf_cuda_lib
bear -- make
cd ..
jq -s 'map(.[])' gmf_{c,cuda}_lib/compile_commands.json > compile_commands.json
```


### linting with flake8

To lint a specific file using this projects setup, run

```bash
flake8 --config setup.cfg ./src/hardtarget/file.py
```

or on a entire folder

```bash
flake8 --config setup.cfg ./src/hardtarget/tools/
```

#### flake8 linting with Code (VS / OSS)

Open command palette and set the python interpreter to your local environment 
by searching for `Python: select interpreter`. 

Then select flake8 as the linter by searching for `Python: select linter` in 
the command palette. 

### formatting with black

To format a single file, run

```bash
black --config pyproject.toml ./src/hardtarget/file.py
```

or target an entire folder with 

```bash
black --config pyproject.toml ./src/hardtarget/tools/
```

#### To enable auto-formatting in Code (VS / OSS)

Add the black provider to Code's config with

```json
"python.formatting.provider": "black",
```

Then auto-formatting of the current file is by default bound to `Ctrl+Shift+i` 
and can be changed by searching for the keybinding `Format document`.

### pytest

To run entire suite
```bash
pytest
```
To run specific file
```bash
pytest tests/test_gmf.py
```

To run specific TestCase, within file
```bash
pytest tests/test_gmf.py::TestGMF
```

To run specific test, within TestCase, within file.
```bash
pytest tests/test_gmf.py::TestGMF::test_gmf
```

To run cuda tests (skipped by default).
```bash
pytest -m cuda
```
