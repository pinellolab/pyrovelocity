"""Tests for `pyrovelocity.tasks.summarize` module."""


import pytest


def test_load_summarize():
    from pyrovelocity.tasks import summarize

    print(summarize.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_summarize_dataset(summarize_dataset_output):
    return summarize_dataset_output


@pytest.mark.slow
@pytest.mark.integration
def test_summarize_dataset_model1(summarize_dataset_model1_output):
    return summarize_dataset_model1_output
