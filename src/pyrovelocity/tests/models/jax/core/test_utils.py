"""Tests for PyroVelocity JAX/NumPyro core utility functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models.jax.core.utils import (
    create_key,
    split_key,
    check_array_shape,
    check_array_dtype,
    ensure_array,
)


def test_create_key():
    """Test create_key function."""
    key = create_key(42)
    assert isinstance(key, jnp.ndarray)
    assert key.shape == (2,)


def test_split_key(jax_key):
    """Test split_key function."""
    keys = split_key(jax_key)
    assert len(keys) == 2
    assert isinstance(keys[0], jnp.ndarray)
    assert isinstance(keys[1], jnp.ndarray)
    assert keys[0].shape == (2,)
    assert keys[1].shape == (2,)
    assert not jnp.array_equal(keys[0], keys[1])

    # Test with num parameter
    keys = split_key(jax_key, num=3)
    assert len(keys) == 3
    assert all(isinstance(k, jnp.ndarray) for k in keys)
    assert all(k.shape == (2,) for k in keys)


def test_create_key_type_checking():
    """Test create_key type checking."""
    # Valid input
    key = create_key(42)
    assert isinstance(key, jnp.ndarray)
    
    # Invalid input
    with pytest.raises(BeartypeCallHintParamViolation):
        create_key("not_an_int")


def test_split_key_type_checking(jax_key):
    """Test split_key type checking."""
    # Valid input
    key1, key2 = split_key(jax_key)
    assert isinstance(key1, jnp.ndarray)
    assert isinstance(key2, jnp.ndarray)
    
    # Invalid input
    with pytest.raises(BeartypeCallHintParamViolation):
        split_key("not_a_key")


def test_check_array_shape(jax_array_1d, jax_array_2d):
    """Test check_array_shape function."""
    assert check_array_shape(jax_array_1d, (5,))
    assert not check_array_shape(jax_array_1d, (4,))
    assert check_array_shape(jax_array_2d, (2, 3))
    assert not check_array_shape(jax_array_2d, (3, 2))


def test_check_array_dtype(jax_array_1d):
    """Test check_array_dtype function."""
    assert check_array_dtype(jax_array_1d, jnp.float32)
    assert not check_array_dtype(jax_array_1d, jnp.int32)
    
    # Convert to int32 and test
    int_array = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    assert check_array_dtype(int_array, jnp.int32)
    assert not check_array_dtype(int_array, jnp.float32)


def test_ensure_array():
    """Test ensure_array function."""
    # Test with list
    list_input = [1.0, 2.0, 3.0]
    array_output = ensure_array(list_input)
    assert isinstance(array_output, jnp.ndarray)
    assert jnp.array_equal(array_output, jnp.array(list_input))
    
    # Test with tuple
    tuple_input = (1.0, 2.0, 3.0)
    array_output = ensure_array(tuple_input)
    assert isinstance(array_output, jnp.ndarray)
    assert jnp.array_equal(array_output, jnp.array(tuple_input))
    
    # Test with numpy array
    numpy_input = np.array([1.0, 2.0, 3.0])
    array_output = ensure_array(numpy_input)
    assert isinstance(array_output, jnp.ndarray)
    assert jnp.array_equal(array_output, jnp.array(numpy_input))
    
    # Test with JAX array
    jax_input = jnp.array([1.0, 2.0, 3.0])
    array_output = ensure_array(jax_input)
    assert isinstance(array_output, jnp.ndarray)
    assert jnp.array_equal(array_output, jax_input)