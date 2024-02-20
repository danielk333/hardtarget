# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import pathlib
import warnings
from datetime import date
import hardtarget

docs_src_path = pathlib.Path(__file__).parent.resolve()
ext_paths = docs_src_path / "extensions"
sys.path.append(str(ext_paths))

# -- Project information -----------------------------------------------------

project = "hardtarget"
version = ".".join(hardtarget.__version__.split(".")[:2])
release = hardtarget.__version__

copyright = f"2023-{date.today().year}, Daniel Kastinen, Juha Vierinen, et al."
author = "Daniel Kastinen, Juha Vierinen, et al."


# -- General configuration ---------------------------------------------------

add_module_names = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "sphinx_c_autodoc",
    "myst_nb",
    "irf.autopackages",
    "sphinx_gallery.load_style",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# autopackages settings
irf_autopackages_toctree = "autopackages"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*.ipynb", "examples"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".myst": "myst-nb",
}

# The master toctree document.
master_doc = "index"

# MyST-NB config
myst_enable_extensions = [
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "basic"
html_favicon = "static/favicon.png"
html_logo = "static/logo.png"
html_css_files = [
    "fixes.css",
    "https://www.irf.se/branding/irf.css",
    "https://www.irf.se/branding/irf-sphinx-basic.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]


# -- Options for gallery extension ---------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "autogallery",  # path where to save gallery generated examples
    "filename_pattern": r".*\.py",
    "ignore_pattern": r".*__no_agl\.py",
}

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# -----------------------------------------------------------------------------
# c/c++ docs
# -----------------------------------------------------------------------------

c_autodoc_roots = [
    str(docs_src_path.parents[1] / "src" / "gmf_c_lib"),
    str(docs_src_path.parents[1] / "src" / "gmf_cuda_lib"),
]
c_autodoc_compilation_database = str(docs_src_path.parents[1] / "src" / "compile_commands.json")


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False


# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("http://matplotlib.sourceforge.net/", None),
}
