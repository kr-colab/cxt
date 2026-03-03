# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'cxt'
copyright = '2025, Kevin Korfmann | Logo designed by Negar Rahnamae'
author = 'Kevin Korfmann'

version = ''
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns = []
pygments_style = None

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

html_static_path = ['_static']

htmlhelp_basename = 'cxtdoc'

html_logo = "./figures/logo_3d_2.png"

html_theme_options = {
    'sidebar_hide_name': True,
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc, 'cxt.tex', 'cxt Documentation',
     'Kevin Korfmann', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'cxt', 'cxt Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'cxt', 'cxt Documentation',
     author, 'cxt',
     'Transformer-based pairwise coalescent time inference.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_exclude_files = ['search.html']
