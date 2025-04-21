"""Tests for preprocessing utilities."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrovelocity.models.jax.data.preprocessing import (
    normalize_counts,
    compute_size_factors,
    filter_genes,
    _internal_transform,
)


@pytest.fixture
def mock_count_data():
    """Create mock count data for testing."""
    # Create random count data
    np.random.seed(42)
    u_counts = np.random.poisson(5, size=(20, 100))
    s_counts = np.random.poisson(8, size=(20, 100))

    # Convert to JAX arrays
    u_counts_jax = jnp.array(u_counts)
    s_counts_jax = jnp.array(s_counts)

    return u_counts_jax, s_counts_jax


def test_normalize_counts(mock_count_data):
    """Test normalize_counts function."""
    u_counts, _ = mock_count_data

    # Test with default parameters
    # Provide explicit size factors to avoid broadcasting issues
    size_factors = jnp.ones(u_counts.shape[0])
    normalized = normalize_counts(u_counts, size_factors=size_factors)

    # Check that the returned object is a JAX array
    assert isinstance(normalized, jnp.ndarray)

    # Check that the array has the same shape as the input
    assert normalized.shape == u_counts.shape

    # Check that the values are log-transformed (should be smaller than original)
    assert jnp.mean(normalized) < jnp.mean(u_counts)

    # Test without log transform
    normalized_no_log = normalize_counts(u_counts, log_transform=False)

    # Check that the values are not log-transformed
    assert jnp.mean(normalized_no_log) > jnp.mean(normalized)

    # Test with custom size factors
    size_factors = jnp.ones(u_counts.shape[0]) * 2  # Double the size factors
    normalized_custom_sf = normalize_counts(u_counts, size_factors=size_factors)

    # Check that the values are smaller with larger size factors
    assert jnp.mean(normalized_custom_sf) < jnp.mean(normalized)

    # Test with custom pseudocount
    normalized_custom_pc = normalize_counts(u_counts, pseudocount=0.1)

    # Check that the values are different with a different pseudocount
    assert not jnp.allclose(normalized, normalized_custom_pc)


def test_compute_size_factors(mock_count_data):
    """Test compute_size_factors function."""
    u_counts, s_counts = mock_count_data

    # Test with unspliced counts
    u_size_factors = compute_size_factors(u_counts, axis=1)

    # Check that the returned object is a JAX array
    assert isinstance(u_size_factors, jnp.ndarray)

    # Check that the array has the correct shape
    assert u_size_factors.shape == (u_counts.shape[0],)

    # Check that the size factors are positive
    assert jnp.all(u_size_factors > 0)

    # Test with spliced counts
    s_size_factors = compute_size_factors(s_counts)

    # Check that the size factors are different for different inputs
    assert not jnp.allclose(u_size_factors, s_size_factors)

    # Test with different axis
    gene_size_factors = compute_size_factors(u_counts, axis=0)

    # Check that the array has the correct shape
    assert gene_size_factors.shape == (u_counts.shape[1],)


def test_filter_genes(mock_count_data):
    """Test filter_genes function."""
    u_counts, s_counts = mock_count_data

    # Test with default parameters
    u_filtered, s_filtered, gene_indices = filter_genes(u_counts, s_counts)

    # Check that the returned objects are JAX arrays
    assert isinstance(u_filtered, jnp.ndarray)
    assert isinstance(s_filtered, jnp.ndarray)
    assert isinstance(gene_indices, jnp.ndarray)

    # Check that the filtered arrays have the same number of cells
    assert u_filtered.shape[0] == u_counts.shape[0]
    assert s_filtered.shape[0] == s_counts.shape[0]

    # Check that the number of genes is reduced
    assert u_filtered.shape[1] <= u_counts.shape[1]
    assert s_filtered.shape[1] <= s_counts.shape[1]

    # Check that the filtered arrays have the same shape
    assert u_filtered.shape == s_filtered.shape

    # Check that gene_indices has the correct shape
    assert gene_indices.shape == (u_filtered.shape[1],)

    # Test with stricter filtering parameters
    u_filtered_strict, s_filtered_strict, gene_indices_strict = filter_genes(
        u_counts, s_counts, min_counts=20, min_cells=10
    )

    # Check that stricter filtering results in fewer genes
    assert u_filtered_strict.shape[1] <= u_filtered.shape[1]

    # Test with parameters that should filter out all genes
    u_filtered_all, s_filtered_all, gene_indices_all = filter_genes(
        u_counts, s_counts, min_counts=1000, min_cells=1000
    )

    # Check that the arrays have 0 genes
    assert u_filtered_all.shape[1] == 0
    assert s_filtered_all.shape[1] == 0
    assert gene_indices_all.shape[0] == 0


def test_internal_transform():
    """Test _internal_transform function."""
    # Create test data
    data = {
        "X_spliced": jnp.array([[-1.0, 2.0], [3.0, -4.0]]),
        "X_unspliced": jnp.array([[0.0, -2.0], [-3.0, 4.0]]),
        "other_key": jnp.array([1.0, 2.0]),
    }

    # Call the function
    transformed = _internal_transform(data)

    # Check that the returned object is a dictionary
    assert isinstance(transformed, dict)

    # Check that the dictionary has the same keys as the input
    assert set(transformed.keys()) == set(data.keys())

    # Check that count matrices have non-negative values
    assert jnp.all(transformed["X_spliced"] >= 0)
    assert jnp.all(transformed["X_unspliced"] >= 0)

    # Check that other keys are unchanged
    assert jnp.array_equal(transformed["other_key"], data["other_key"])

    # Check that zero values are preserved and negative values are set to zero
    assert transformed["X_spliced"][0, 0] == 0
    assert transformed["X_unspliced"][0, 0] == 0


def test_normalize_counts_edge_cases():
    """Test normalize_counts function with edge cases."""
    # Test with zeros
    zeros = jnp.zeros((5, 10))
    # Provide explicit size factors to avoid broadcasting issues
    size_factors = jnp.ones(5)
    normalized_zeros = normalize_counts(zeros, size_factors=size_factors)

    # Check that zeros remain zeros after normalization
    assert jnp.all(normalized_zeros == 0)

    # Test with ones
    ones = jnp.ones((5, 10))
    size_factors_ones = jnp.ones(5)
    normalized_ones = normalize_counts(ones, size_factors=size_factors_ones)

    # Check that ones become log(1 + pseudocount) after normalization
    expected = jnp.log(1 + 1.0)  # Default pseudocount is 1.0
    assert jnp.allclose(normalized_ones, expected)

    # Test with NaNs
    data_with_nans = jnp.array([[1.0, 2.0], [jnp.nan, 4.0]])
    normalized_nans = normalize_counts(data_with_nans)

    # Check that NaNs remain NaNs after normalization
    assert jnp.isnan(normalized_nans[1, 0])
    assert not jnp.isnan(normalized_nans[0, 0])
    assert not jnp.isnan(normalized_nans[0, 1])
    assert not jnp.isnan(normalized_nans[1, 1])


def test_compute_size_factors_edge_cases():
    """Test compute_size_factors function with edge cases."""
    # Test with zeros
    zeros = jnp.zeros((5, 10))
    size_factors_zeros = compute_size_factors(zeros, axis=1)

    # Check that size factors are ones when counts are zeros
    assert size_factors_zeros.shape == (5,)
    assert jnp.all(size_factors_zeros == 1.0)

    # Test with ones
    ones = jnp.ones((5, 10))
    size_factors_ones = compute_size_factors(ones)

    # Check that size factors are ones when all counts are equal
    assert jnp.allclose(size_factors_ones, jnp.ones(5))
