"""Tests for `pyrovelocity.utils` module."""

import hashlib
from pathlib import Path

import hypothesis
import numpy as np
import pytest
from anndata import AnnData
from hypothesis import given
from hypothesis import strategies as st

from pyrovelocity.utils import generate_sample_data, hash_file


def test_load_utils():
    from pyrovelocity import utils

    print(utils.__file__)


n_obs_strategy = st.integers(min_value=5, max_value=100)
n_vars_strategy = st.integers(min_value=5, max_value=100)
rate_strategy = st.floats(min_value=0.01, max_value=3.0)
beta_gamma_strategy = st.tuples(
    st.floats(min_value=0.01, max_value=3.0),  # beta
    st.floats(min_value=0.01, max_value=3.0),  # gamma
).filter(lambda x: x[1] > x[0])  # gamma required to be larger than beta
noise_model_strategy = st.sampled_from(["iid", "normal"])
random_seed_strategy = st.randoms()


@hypothesis.settings(deadline=1500)
@given(
    n_obs=n_obs_strategy,
    n_vars=n_vars_strategy,
    alpha=rate_strategy,
    beta_gamma=beta_gamma_strategy,
    alpha_=rate_strategy,
    noise_model=noise_model_strategy,
    random_seed=random_seed_strategy,
)
def test_generate_sample_data(
    n_obs: int,
    n_vars: int,
    alpha: float,
    beta_gamma: tuple,
    alpha_: float,
    noise_model: str,
    random_seed: np.random.RandomState,
):
    beta, gamma = beta_gamma
    seed = random_seed.randint(0, 1000)

    adata = generate_sample_data(
        n_obs=n_obs,
        n_vars=n_vars,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        alpha_=alpha_,
        noise_model=noise_model,
        random_seed=seed,
    )

    assert isinstance(adata, AnnData)
    assert adata.shape == (n_obs, n_vars)
    assert "spliced" in adata.layers
    assert "unspliced" in adata.layers

    new_seed = seed
    while new_seed == seed:
        new_seed = random_seed.randint(0, 1000)

    # Check that different seeds produce different results
    adata2 = generate_sample_data(
        n_obs=n_obs,
        n_vars=n_vars,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        alpha_=alpha_,
        noise_model=noise_model,
        random_seed=new_seed,
    )

    assert not (adata.X == adata2.X).all()


@pytest.mark.parametrize("n_obs, n_vars", [(100, 12), (50, 10), (200, 20)])
@pytest.mark.parametrize("noise_model", ["iid", "gillespie", "normal"])
def test_generate_sample_data_dimensions(n_obs, n_vars, noise_model):
    adata = generate_sample_data(
        n_obs=n_obs, n_vars=n_vars, noise_model=noise_model, random_seed=98
    )
    assert adata.shape == (n_obs, n_vars)


def test_generate_sample_data_layers(default_sample_data):
    assert "spliced" in default_sample_data.layers
    assert "unspliced" in default_sample_data.layers


def test_generate_sample_data_reproducibility(default_sample_data):
    adata1 = default_sample_data
    adata2 = generate_sample_data(random_seed=98)
    assert (adata1.X == adata2.X).all()
    assert (adata1.layers["spliced"] == adata2.layers["spliced"]).all()
    assert (adata1.layers["unspliced"] == adata2.layers["unspliced"]).all()


def test_generate_sample_data_invalid_noise_model():
    with pytest.raises(
        ValueError,
        match="noise_model must be one of 'iid', 'gillespie', 'normal'",
    ):
        generate_sample_data(noise_model="wishful thinking")


def create_file_with_content(path: Path, content: str):
    path.write_text(content)
    return path


def calculate_sha256(content: str):
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.fixture
def sample_files(tmp_path):
    files = {
        "empty": create_file_with_content(tmp_path / "empty.txt", ""),
        "small": create_file_with_content(tmp_path / "small.txt", "test file"),
        "medium": create_file_with_content(
            tmp_path / "medium.txt", "A" * 10000
        ),
        "large": create_file_with_content(
            tmp_path / "large.txt", "B" * 1000000
        ),
    }
    return files


def test_hash_file(sample_files):
    for file_type, file_path in sample_files.items():
        content = file_path.read_text()
        expected_hash = calculate_sha256(content)
        assert (
            hash_file(file_path) == expected_hash
        ), f"Hash mismatch for {file_type} file"


def test_hash_file_nonexistent_file(tmp_path):
    non_existent_file = tmp_path / "nonexistent.txt"
    with pytest.raises(FileNotFoundError):
        hash_file(non_existent_file)


def test_hash_file_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        hash_file(tmp_path)
