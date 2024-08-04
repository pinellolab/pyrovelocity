"""Tests for `pyrovelocity.tasks.train` module."""

import pytest


def test_load_train():
    from pyrovelocity.tasks import train

    print(train.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_train_dataset(train_dataset_output):
    return train_dataset_output


@pytest.mark.slow
@pytest.mark.integration
def test_train_dataset_model1(train_dataset_model1_output):
    return train_dataset_model1_output
