"""Tests for `pyrovelocity.tasks.summarize` module."""


import pytest


def test_load_summarize():
    from pyrovelocity.tasks import summarize

    print(summarize.__file__)


@pytest.mark.slow
def test_summarize_dataset(summarize_dataset_output):
    return summarize_dataset_output

