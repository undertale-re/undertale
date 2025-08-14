# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import undertale

# -- Project information -----------------------------------------------------

project = undertale.__title__
copyright = undertale.__copyright__
author = undertale.__author__
release = undertale.__version__

# -- General configuration ---------------------------------------------------

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_favicon = "favicon.png"
html_theme_options = {
    "show_relbar_bottom": True,
}
html_show_sphinx = False

# -- Extension configuration -------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": None,
    "members": None,
}

# -- Custom modifications ----------------------------------------------------
