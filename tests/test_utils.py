"""Tests for `pyrovelocity.utils` module."""

import pytest
from pyrovelocity.utils import generate_sample_data


def test_load_utils():
    from pyrovelocity import utils

    print(utils.__file__)


@pytest.fixture
def sample_data():
    return generate_sample_data(random_seed=42)


@pytest.mark.parametrize("n_obs, n_vars", [(100, 12), (50, 10), (200, 20)])
@pytest.mark.parametrize("noise_model", ["iid", "gillespie", "normal"])
def test_generate_sample_data_dimensions(n_obs, n_vars, noise_model):
    adata = generate_sample_data(
        n_obs=n_obs, n_vars=n_vars, noise_model=noise_model, random_seed=42
    )
    assert adata.shape == (n_obs, n_vars)

