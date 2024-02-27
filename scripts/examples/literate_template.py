# ---
# jupyter:
#   celltoolbar: Slideshow
#   jupytext:
#     cell_metadata_filter: all
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     name: python
#   rise:
#     scroll: true
#     theme: black
#   toc-autonumbering: true
#   toc-showcode: false
#   toc-showmarkdowntxt: false
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Notebook template

# %% [markdown] slideshow={"slide_type": "fragment"}
# Use this file as a template for creating new notebooks.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Dependencies

# %% [markdown] slideshow={"slide_type": "fragment"}
# Copies of this template will require the following dependencies:
#
# - jupytext for converting between notebook and script formats
# - jupyter for nbconvert
# - junix for extracting figures
#
# Its other dependencies are not specific to the purpose of the template.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Usage

# %% [markdown] slideshow={"slide_type": "fragment"}
# Initialize the notebook with the following code:
#
# ```bash
# jupytext --set-formats ipynb,py:percent literate_template.py
# ```
#
# Synchronize the python file with an ipython notebook file using
#
# ```bash
# jupytext --sync literate_template.py
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Setup

# %% slideshow={"slide_type": "fragment"}
import os

import matplotlib.pyplot as plt
import numpy as np


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Parameters

# %% [markdown] slideshow={"slide_type": "fragment"}
# The following cell defines parameters to be exposed to papermill execution.

# %% slideshow={"slide_type": "fragment"} tags=["parameters"]
TEST_MODE = False

# %% [markdown] slideshow={"slide_type": "fragment"}
# Parameters injected by papermill should appear above.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Functions

# %% [markdown] slideshow={"slide_type": "fragment"}
# Define functions to be used in the notebook.


# %% slideshow={"slide_type": "fragment"}
def plot_random_numbers(size=10):
    plt.plot(np.random.rand(size))
    plt.show()


# %% [markdown] slideshow={"slide_type": "slide"}
# This cell checks if the notebook is running in test mode.
# If it is, it will reduce the size of the data to be processed.

# %% slideshow={"slide_type": "fragment"}
print(f"Test mode: {TEST_MODE}")
size = 5 if TEST_MODE else 10
print(f"Size: {size}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Results

# %% [markdown] slideshow={"slide_type": "fragment"}
# Here are the results of the analysis.

# %% slideshow={"slide_type": "fragment"}
plot_random_numbers(size)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Postprocessing

# %% [markdown] slideshow={"slide_type": "fragment"}
# The notebook can be converted to html, pdf, or slides using nbconvert:
#
# ```bash
# jupyter nbconvert --to html --template lab --theme dark literate_template.ipynb
# jupyter nbconvert --to pdf literate_template.ipynb
# jupyter nbconvert --to slides literate_template.ipynb --post serve
# ```
#
# figures can be extracted using junix:
#
# ```bash
# mkdir -p figures
# junix -f literate_template.ipynb -o figures/ -p figure
# ```
