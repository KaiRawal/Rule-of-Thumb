"""Sphinx configuration for the ruleofthumb docs (hosted on ReadTheDocs)."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "ruleofthumb"
author = "RoT authors"
version = "0.0.1"
release = "0.0.1"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = [".md", ".ipynb"]
exclude_patterns = ["_build"]

myst_enable_extensions = ["colon_fence", "deflist"]

nb_execution_mode = "auto"
nb_execution_timeout = 600

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "ruleofthumb"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "github_url": "https://github.com/KaiRawal/Rule-of-Thumb",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "announcement": "⚠️ Experimental 0.0.x pre-alpha — expect breaking changes before 1.0.",
}
html_context = {
    "github_user": "KaiRawal",
    "github_repo": "Rule-of-Thumb",
    "github_version": "main",
    "doc_path": "docs-src",
}
