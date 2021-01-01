"""Sphinx configuration."""
from datetime import datetime

project = "Simple Hierarchy"
author = "Rajiv Sarvepalli"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]
html_static_path = ["_static"]
copybutton_prompt_text = "$ "
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/rajivsarvepalli/SimpleHierarchy",
    "show_prev_next": False,
}

html_css_files = [
    "css/getting_started.css",
    "css/pandas.css",
]
version = "1.0.1"
