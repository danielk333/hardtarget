import os
from os.path import join as pjoin
import codecs
import pathlib

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).resolve().parents[0]


def find_in_path(name, path):
    """Find a file in a search path"""
    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """
    # Adapted from https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py

    # First check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDAHOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError("The CUDA %s path could not be " "located in %s" % (k, v))

    return cudaconfig


cudasources = ["src/gmf_cuda_lib/gmfgpu.cu"]
cudalibraries = ["cufft"]

csources = ["src/gmf_c_lib/gmf.c"]
clibraries = ["fftw3f"]

# Build C module
gmfcmodule = Extension(
    # Where to store the .so file
    name="hardtarget.gmf.gmfclib",
    # Libraries used
    libraries=clibraries,
    extra_compile_args={
        "gcc": [],
    },
    # Path to cuda source files, relative to repo root
    sources=csources,
)

# Try to build Cuda module
try:
    CUDA = locate_cuda()
    print(CUDA)
    print("Cuda compiler detected, compiling GPU optimized code")

    gmfgpumodule = Extension(
        # Where to store the .so file
        name="hardtarget.gmf.gmfcudalib",
        library_dirs=[CUDA["lib64"]],
        # Libraries used
        libraries=cudalibraries,
        extra_compile_args={
            "gcc": [],
            "nvcc": [
                "-O3",
                "--compiler-options",
                "'-fPIC'",
            ],
        },
        include_dirs=[CUDA["include"]],
        # Path to cuda source files, relative to repo root
        sources=cudasources,
    )
    # Specify that we want both C and Cuda module to be compiled
    # The C module has to be built first because of the way we inject
    # a new linker without cleaning up afterwards
    extmodules = [gmfcmodule, gmfgpumodule]
except EnvironmentError:
    print("No Cuda compiler detected, compiling serial code")
    # If on cuda environment exists then only compile C code
    extmodules = [gmfcmodule]


def customize_compiler_for_nvcc(self):
    # Adapted from https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py

    """
    Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    # default_linker_so = self.linker_so
    superCompile = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            self.set_executable("linker_so", [CUDA["nvcc"], "--shared"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        superCompile(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so and linker_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    # Adapted from https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def get_version(path):
    with codecs.open(path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


setup(
    cmdclass={"build_ext": custom_build_ext},
    ext_modules=extmodules,
    version=get_version(HERE / "src" / "hardtarget" / "version.py"),
)
