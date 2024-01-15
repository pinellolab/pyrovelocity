"""
bazel macros for python
"""

load("@bazel_skylib//lib:paths.bzl", "paths")

def py_test_module_list(files, size, deps, extra_srcs=[], name_suffix="", **kwargs):
    """
    py_test_module_list creates a py_test for each file in files.
    """
    for file in files:
        # remove .py
        name = paths.split_extension(file)[0] + name_suffix
        if name == file:
            basename = basename + "_test"
        native.py_test(
            name = name,
            size = size,
            main = file,
            srcs = extra_srcs + [file],
            deps = deps,
            **kwargs
        )
