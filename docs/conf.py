"""Sphinx configuration."""
from __future__ import absolute_import, print_function, unicode_literals

import inspect
import os
import sys
from datetime import datetime
from typing import Any, Dict

import simple_hierarchy

project = "Simple Hierarchy"
author = "Rajiv Sarvepalli"
copyright = f"{datetime.now().year}, {author}"
version = simple_hierarchy.__version__
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]
html_static_path = ["_static"]
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/rajivsarvepalli/SimpleHierarchy",
    "show_prev_next": False,
}

html_css_files = [
    "css/getting_started.css",
    "css/pandas.css",
]


def linkcode_resolve(domain: str, info: Dict) -> str:
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    ret = _help_resolve(submod, fullname)
    if ret is None:
        return ret
    else:
        obj, fn = ret
    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    tag_or_branch = os.getenv("SPHINX_MULTIVERSION_NAME", default="master")
    fn = os.path.relpath(fn, start=os.path.dirname(simple_hierarchy.__file__)).replace(
        os.sep, "/"
    )
    url = (
        "https://github.com/rajivsarvepalli/SimpleHierarchy"
        "/blob/%s/src/SimpleHierarchy/%s%s"
    )
    url = url % (tag_or_branch, fn, linespec)
    return url


def _help_resolve(submod: Any, fullname: str) -> Any:
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    return obj, fn
