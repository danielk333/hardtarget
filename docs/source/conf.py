# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date
import hardtarget

project = "hardtarget"
version = ".".join(hardtarget.__version__.split(".")[:2])
release = hardtarget.__version__
copyright = f"2023-{date.today().year}, Daniel Kastinen, Juha Vierinen, et al."
author = "Daniel Kastinen, Juha Vierinen, et al."


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex"
]

# Path to your bibliography files
bibtex_bibfiles = ['references.bib']

html_favicon = "_static/favicon.png"
html_logo = "_static/logo.png"

templates_path = ["_templates"]
exclude_patterns = []

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "style_nav_header_background": "black",
}

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False


# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': False,
    'show-inheritance': False,
}

