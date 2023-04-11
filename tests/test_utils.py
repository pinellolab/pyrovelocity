"""Tests for `pyrovelocity.utils` module."""

import pytest
from pyrovelocity.utils import generate_sample_data


def test_load_utils():
    from pyrovelocity import utils

    print(utils.__file__)


@pytest.fixture
def sample_data():
    return generate_sample_data(random_seed=98)


@pytest.mark.parametrize("n_obs, n_vars", [(100, 12), (50, 10), (200, 20)])
@pytest.mark.parametrize("noise_model", ["iid", "gillespie", "normal"])
def test_generate_sample_data_dimensions(n_obs, n_vars, noise_model):
    adata = generate_sample_data(
        n_obs=n_obs, n_vars=n_vars, noise_model=noise_model, random_seed=98
    )
    assert adata.shape == (n_obs, n_vars)


def test_generate_sample_data_layers(sample_data):
    assert "spliced" in sample_data.layers
    assert "unspliced" in sample_data.layers


def test_generate_sample_data_reproducibility():
    adata1 = generate_sample_data(random_seed=98)
    adata2 = generate_sample_data(random_seed=98)
    assert (adata1.X == adata2.X).all()
    assert (adata1.layers["spliced"] == adata2.layers["spliced"]).all()
    assert (adata1.layers["unspliced"] == adata2.layers["unspliced"]).all()

