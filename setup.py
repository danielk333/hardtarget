import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#Haven't implemented this possibility yet
cuda = False
#Compile C code into shared library
if (cuda):
    #Compile Cuda code
    pass
else:
    gmfmodule = setuptools.Extension(
                    #Where to store the .so file
                    'hardtarget.libgmf',
                    #Libraries used
                    libraries=['fftw3f'],
                    #Path to C source files, relative to repo root
                    sources=['src/hardtarget/cfiles/gmf.c'],
                    )
                    
setuptools.setup(
    name="hardtarget",
    version="0.0.7",
    author="Juha Vierinen",
    author_email="juha-pekka.vierinen@uit.no",
    description="Hard target processing of radar data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: Sphinx",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
    ],
    install_requires=[
        'h5py>=3.2.1',
        'digital_rf>=2.6.6',
        'numpy>=1.20.3',
        'scipy>=1.4.1',
        'matplotlib'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    ext_modules=[gmfmodule]
)