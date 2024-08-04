"""Tests for `pyrovelocity.tasks.postprocess` module."""

import pytest


def test_load_postprocess():
    from pyrovelocity.tasks import postprocess

    print(postprocess.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_postprocess_dataset(postprocess_dataset_output):
    return postprocess_dataset_output


@pytest.mark.slow
@pytest.mark.integration
def test_postprocess_dataset_model1(postprocess_dataset_model1_output):
    return postprocess_dataset_model1_output
