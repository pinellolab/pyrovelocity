"""
bazel macros for python
"""

# load("@aspect_rules_py//py:defs.bzl", "py_pytest_main", "py_test")
load("@bazel_skylib//lib:paths.bzl", "paths")
# load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")



def py_test_module_list(files, size, deps, args=[], extra_srcs=[], name_suffix="", **kwargs):
    """
    py_test_module_list creates a py_test for each file in files.    
    """
    for file in files:
        name = paths.split_extension(file)[0] + name_suffix
        if name == file:
            name = name + "_test"
        # native.py_test(
        py_test(
            name = name,
            size = size,
            main = file,
            srcs = extra_srcs + [file],
            args = [
                "-rA",
                "-v",
                "--disable-warnings",
            ] + args,
            deps = deps,
            **kwargs
        )

def xdoctest(files, name="xdoctest", deps=[], srcs=[], data=[], args=[], size="medium", tags=[], **kwargs):
    """
    run xdoctests on library files
    """
    files = native.glob(include=files, exclude=["__init__.py"])

    # native.py_test(
    py_test(
        name = name,
        srcs = ["//bazel:pytest_wrapper.py"] + srcs,
        main = "//bazel:pytest_wrapper.py",
        size = size,
        args = [
            "-rA",
            "-v",
            "--disable-warnings",
            "--xdoctest",
            "-c=$(location //bazel:conftest.py)",
        ] + args + ["$(location :%s)" % file for file in files],
        data = ["//bazel:conftest.py"] + files + data,
        # python_version = "PY3",
        # srcs_version = "PY3",
        tags = tags,
        deps = deps,
        **kwargs
    )

# for use with aspect_rules_py 
# def py_test_module_list(files, size, deps, extra_srcs=[], name_suffix="", **kwargs):
#     """
#     py_test_module_list creates a py_test for each file in files.    
#     """
#     py_pytest_main(
#         name = "__test__",
#         deps = [requirement("pytest")],
#     )
#     for file in files:
#         name = paths.split_extension(file)[0] + name_suffix
#         if name == file:
#             name = name + "_test"
#         py_test(
#             name = name,
#             size = size,
#             main = ":__test__.py",
#             srcs = extra_srcs + [file] + [":__test__"],
#             deps = deps + [":__test__", requirement("pytest")],
#             **kwargs
#         )
