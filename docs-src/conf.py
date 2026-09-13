"""Sphinx configuration for the ruleofthumb docs (hosted on ReadTheDocs)."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from ruleofthumb import __version__

project = "ruleofthumb"
author = "RoT authors"
version = __version__
release = __version__

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
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "github_url": "https://github.com/KaiRawal/Rule-of-Thumb",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": True,
    "navbar_align": "left",
    "show_prev_next": True,
    "back_to_top_button": True,
    "search_as_you_type": True,
    "pygments_light_style": "a11y-high-contrast-light",
    "pygments_dark_style": "a11y-high-contrast-dark",
    "announcement": "⚠️ Experimental 0.0.x pre-alpha — expect breaking changes before 1.0.",
    "show_version_warning_banner": True,
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo-dark.svg",
        "text": "ruleofthumb",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/ruleofthumb-rot/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
        {
            "name": "Paper",
            "url": "https://arxiv.org/abs/2608.10766",
            "icon": "fa-solid fa-file-lines",
            "type": "fontawesome",
        },
    ],
    "footer_start": ["copyright", "sphinx-version"],
    "footer_end": ["theme-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "header_links_before_dropdown": 4,
}
html_context = {
    "github_user": "KaiRawal",
    "github_repo": "Rule-of-Thumb",
    "github_version": "main",
    "doc_path": "docs-src",
}
