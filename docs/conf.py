project = "IVAC"
author = "Chatipat Lorpaiboon"
year = "2020"
copyright = "{}, {}".format(year, author)
version = release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"
html_short_title = "{}-{}".format(project, version)
html_sidebars = {"**": ["searchbox.html", "globaltoc.html"]}

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
