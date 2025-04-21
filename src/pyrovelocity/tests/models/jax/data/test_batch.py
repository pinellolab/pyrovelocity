"""Tests for batch processing utilities."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pyrovelocity.models.jax.data.batch import (
    random_batch_indices,
    create_batch_iterator,
    batch_data,
    vmap_batch_function,
)


@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    key = jax.random.PRNGKey(0)

    # Create random data
    data = {
        "X": jnp.array(np.random.normal(size=(100, 10))),
        "y": jnp.array(np.random.normal(size=(100, 1))),
        "z": jnp.array(np.random.normal(size=(100, 5))),
    }

    return key, data


def test_random_batch_indices():
    """Test random_batch_indices function."""
    key = jax.random.PRNGKey(0)
    num_items = 100
    batch_size = 32

    # Call the function
    indices = random_batch_indices(key, num_items, batch_size)

    # Check that the returned object is a JAX array
    assert isinstance(indices, jnp.ndarray)

    # Check that the array has the correct shape
    assert indices.shape == (batch_size,)

    # Check that the indices are within the valid range
    assert jnp.all(indices >= 0)
    assert jnp.all(indices < num_items)

    # Check that the indices are integers
    assert indices.dtype == jnp.int32 or indices.dtype == jnp.int64

    # Check that different keys produce different indices
    key2 = jax.random.PRNGKey(1)
    indices2 = random_batch_indices(key2, num_items, batch_size)
    assert not jnp.array_equal(indices, indices2)


def test_create_batch_iterator(mock_data):
    """Test create_batch_iterator function."""
    key, data = mock_data
    batch_size = 32

    # Call the function
    iterator = create_batch_iterator(data, batch_size, key)

    # Check that the returned object is an iterator
    assert hasattr(iterator, "__iter__")
    assert hasattr(iterator, "__next__")

    # Check that the iterator produces the expected number of batches
    batches = list(iterator)
    expected_num_batches = (data["X"].shape[0] + batch_size - 1) // batch_size
    assert len(batches) == expected_num_batches

    # Check that each batch has the correct structure
    for batch in batches:
        assert isinstance(batch, dict)
        assert set(batch.keys()) == set(data.keys())

        # Check that each array in the batch has the correct first dimension
        for k in batch:
            if batch is not batches[-1] or data[k].shape[0] % batch_size == 0:
                assert batch[k].shape[0] == batch_size
            else:
                # Last batch might be smaller
                assert batch[k].shape[0] == data[k].shape[0] % batch_size

            # Check that the rest of the dimensions match
            assert batch[k].shape[1:] == data[k].shape[1:]

    # Check that with shuffle=False, the batches are deterministic
    iterator1 = create_batch_iterator(data, batch_size, key, shuffle=False)
    iterator2 = create_batch_iterator(data, batch_size, key, shuffle=False)

    batches1 = list(iterator1)
    batches2 = list(iterator2)

    for b1, b2 in zip(batches1, batches2):
        for k in b1:
            assert jnp.array_equal(b1[k], b2[k])


def test_batch_data(mock_data):
    """Test batch_data function."""
    _, data = mock_data

    # Create some indices
    indices = jnp.array([0, 10, 20, 30, 40])

    # Call the function
    batched = batch_data(data, indices)

    # Check that the returned object is a dictionary
    assert isinstance(batched, dict)

    # Check that the dictionary has the same keys as the input data
    assert set(batched.keys()) == set(data.keys())

    # Check that each array in the batch has the correct shape
    for k in batched:
        assert batched[k].shape[0] == indices.shape[0]
        assert batched[k].shape[1:] == data[k].shape[1:]

    # Check that the batch contains the correct data
    for k in batched:
        expected = data[k][indices]
        assert jnp.array_equal(batched[k], expected)


def test_vmap_batch_function():
    """Test vmap_batch_function function."""

    # Define a simple function to apply to each element in a batch
    def square_sum(x, y):
        return x**2 + y**2

    # Create some test data
    batch_size = 10
    x = jnp.ones((batch_size, 3))
    y = jnp.ones((batch_size, 3)) * 2

    # Apply vmap_batch_function
    batched_fn = vmap_batch_function(square_sum)

    # Call the batched function
    result = batched_fn(x, y)

    # Check that the result has the correct shape
    assert result.shape == (batch_size, 3)

    # Check that the result contains the correct values
    expected = x**2 + y**2
    assert jnp.allclose(result, expected)

    # Test with different in_axes and out_axes
    def weighted_sum(weights, values):
        return jnp.sum(weights * values, axis=-1)

    weights = jnp.ones((batch_size, 5))
    values = jnp.arange(1, 6).reshape(1, 5).repeat(batch_size, axis=0)

    batched_weighted_sum = vmap_batch_function(weighted_sum)
    result = batched_weighted_sum(weights, values)

    # Check that the result has the correct shape (batch_size,)
    assert result.shape == (batch_size,)

    # Check that the result contains the correct values
    expected = jnp.sum(weights * values, axis=-1)
    assert jnp.allclose(result, expected)


def test_vmap_batch_function_with_dict_input():
    """Test vmap_batch_function with dictionary inputs."""

    # Define a function that takes x and y as separate arguments
    def process_xy(x, y):
        return x * y

    # Create some test data
    batch_size = 10
    x = jnp.ones((batch_size, 3))
    y = jnp.ones((batch_size, 3)) * 2

    # Apply vmap_batch_function
    batched_fn = vmap_batch_function(process_xy)

    # Call the batched function
    result = batched_fn(x, y)

    # Check that the result has the correct shape
    assert result.shape == (batch_size, 3)

    # Check that the result contains the correct values
    expected = x * y
    assert jnp.allclose(result, expected)
