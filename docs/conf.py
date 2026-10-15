# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'batbot'
copyright = '2026, Kitware, Inc.'
author = 'Kitware, Inc.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.imgmath',
    'sphinx.ext.napoleon',
    'sphinx_click',
]

suppress_warnings = ['autosectionlabel.*']

# Local/CI builds do not depend on external inventory availability. Enable
# cross-project references deliberately when building with internet access.
intersphinx_mapping = {}
if os.getenv('BATBOT_DOCS_ONLINE') == '1':
    intersphinx_mapping = {
        'python': ('https://docs.python.org/3/', None),
        'numpy': ('https://numpy.org/doc/stable/', None),
    }
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
epub_show_urls = 'footnote'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'colors.rst', 'environment.rst']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}
html_static_path = ['_static']
html_css_files = [
    'theme.css',
]
