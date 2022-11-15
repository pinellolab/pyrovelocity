#!/usr/bin/env python

# In this form, setup.py is a stub to indicate
# this repository contains a python package.
# If you would like to use setuptools
# instead of poetry, a previously functioning
# setup.py for this repository is retained in the
# comments of this file. Otherwise, see pyproject.toml
# and consider managing builds with poetry.

# from setuptools import find_packages
from setuptools import setup


if __name__ == "__main__":
    setup(name="pyrovelocity")

# with open("README.md") as readme_file:
#     readme = readme_file.read()

# with open("docs/history.md") as history_file:
#     history = history_file.read()

# requirements = []

# test_requirements = []

# setup(
#     author="Qian Qin",
#     author_email="qqin@mgh.harvard.edu",
#     python_requires="==3.8",
#     classifiers=[
#         "Development Status :: 2 - Pre-Alpha",
#         "Intended Audience :: Developers",
#         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
#         "Natural Language :: English",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.8",
#     ],
#     description="a multivariate RNA Velocity model to estimate the uncertainty of cell future state using Pyro",
#     install_requires=requirements,
#     license="GNU AFFERO GENERAL PUBLIC LICENSE v3",
#     long_description=readme + "\n\n" + history,
#     include_package_data=True,
#     keywords="pyrovelocity",
#     name="pyrovelocity",
#     packages=find_packages(include=["pyrovelocity", "pyrovelocity.*"]),
#     test_suite="tests",
#     tests_require=test_requirements,
#     url="https://github.com/qinqian/pyrovelocity",
#     version="0.1.0",
#     zip_safe=False,
# )
