"""Tests for `pyrovelocity.tasks.preprocess` module."""
import pytest

from pyrovelocity.tasks.preprocess import preprocess_dataset


def test_load_preprocess():
    from pyrovelocity.tasks import preprocess

    print(preprocess.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_preprocess_dataset(preprocess_dataset_output):
    return preprocess_dataset_output
