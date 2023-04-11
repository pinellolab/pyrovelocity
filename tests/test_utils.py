"""Tests for `pyrovelocity.utils` module."""

import pytest
from pyrovelocity.utils import generate_sample_data


def test_load_utils():
    from pyrovelocity import utils

    print(utils.__file__)


@pytest.fixture
def sample_data():
    return generate_sample_data(random_seed=42)

