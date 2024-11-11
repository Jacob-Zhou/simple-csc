# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import lmcsc

project = 'lmcsc'
copyright = '2024, Houquan Zhou'
author = 'Houquan Zhou'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
            #   'sphinxcontrib.bibtex',
              'sphinx_astrorefs',
              'myst_parser']

templates_path = ['_templates']

source_suffix = ['.rst', '.md']

exclude_patterns = []

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'path_to_docs': 'docs',
    'repository_url': 'https://github.com/Jacob-Zhou/simple-csc',
    'use_edit_page_button': True,
    'use_issues_button': True,
    'use_repository_button': True,
    'use_download_button': True
}
html_copy_source = True

html_static_path = []


autodoc_member_order = 'bysource'

# bibtex
# bibtex_bibfiles = ['refs.bib']