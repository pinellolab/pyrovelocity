"""Tests for `pyrovelocity.io.subset_data` module."""
from pathlib import Path

import pytest

from pyrovelocity.io.subset_data import (
    subset_anndata,
)


def test_subset_data_module():
    from pyrovelocity.io import subset_data

    print(subset_data.__file__)


def test_subset_from_adata(default_sample_data):
    """Test deriving a data subset from an AnnData object."""
    n_obs = 50
    n_vars = 10
    subset_adata, _ = subset_anndata(
        adata=default_sample_data, n_obs=n_obs, n_vars=n_vars
    )
    assert subset_adata.n_obs == n_obs
    assert subset_adata.n_vars == n_vars


def test_subset_from_file(default_sample_data_file):
    """Test deriving a data subset from a file."""
    n_obs = 50
    subset_adata, _ = subset_anndata(
        file_path=default_sample_data_file, n_obs=n_obs
    )
    assert subset_adata.n_obs == n_obs


def test_invalid_input():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError):
        subset_anndata()  # No AnnData object or file path provided


def test_save_subset(default_sample_data, tmp_path):
    """Test saving the subset to a file."""
    n_obs = 50
    output_file = tmp_path / "subset_adata.h5ad"
    _, output_path = subset_anndata(
        adata=default_sample_data,
        n_obs=n_obs,
        save_subset=True,
        output_path=output_file,
    )
    assert output_path == output_file
    assert Path(output_file).exists()


def test_subset_specific_n_obs_vars(default_sample_data):
    """Test deriving a data subset with specific n_obs and n_vars."""
    n_obs = 50
    n_vars = 10
    subset_adata, _ = subset_anndata(
        adata=default_sample_data, n_obs=n_obs, n_vars=n_vars
    )
    assert subset_adata.n_obs == n_obs
    assert subset_adata.n_vars == n_vars
