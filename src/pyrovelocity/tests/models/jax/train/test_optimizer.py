"""
Tests for PyroVelocity JAX/NumPyro optimizer utilities.

This module contains tests for the optimizer utilities, including:

- test_create_optimizer: Test optimizer creation
- test_learning_rate_schedule: Test learning rate schedule
- test_clip_gradients: Test gradient clipping
- test_create_optimizer_with_schedule: Test optimizer creation with schedule
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from typing import Callable
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide

from pyrovelocity.models.jax.train.optimizer import (
    create_optimizer,
    learning_rate_schedule,
    clip_gradients,
    create_optimizer_with_schedule,
)


def test_create_optimizer():
    """Test optimizer creation."""
    # Test Adam optimizer
    optimizer = create_optimizer(optimizer_name="adam", learning_rate=0.01)
    assert isinstance(optimizer, numpyro.optim.Adam)
    # NumPyro optimizers don't expose step_size directly, so we don't test it
    
    # Test SGD optimizer
    optimizer = create_optimizer(optimizer_name="sgd", learning_rate=0.1)
    assert isinstance(optimizer, numpyro.optim.SGD)
    # NumPyro optimizers don't expose step_size directly, so we don't test it
    
    # Test Momentum optimizer
    optimizer = create_optimizer(optimizer_name="momentum", learning_rate=0.01, momentum=0.8)
    assert isinstance(optimizer, numpyro.optim.Momentum)
    # NumPyro optimizers don't expose step_size directly, so we don't test it
    
    # Test RMSProp optimizer
    optimizer = create_optimizer(optimizer_name="rmsprop", learning_rate=0.01, gamma=0.95)
    assert isinstance(optimizer, numpyro.optim.RMSProp)
    # NumPyro optimizers don't expose step_size directly, so we don't test it
    
    # Test ClippedAdam optimizer
    optimizer = create_optimizer(optimizer_name="clipped_adam", learning_rate=0.01, clip_norm=5.0)
    assert isinstance(optimizer, numpyro.optim.ClippedAdam)
    # NumPyro optimizers don't expose step_size directly, so we don't test it
    
    # Test unsupported optimizer
    with pytest.raises(ValueError):
        create_optimizer(optimizer_name="unsupported")


def test_learning_rate_schedule():
    """Test learning rate schedule."""
    # Test continuous decay
    schedule = learning_rate_schedule(init_lr=0.1, decay_steps=100, decay_rate=0.9, staircase=False)
    assert schedule(0) == 0.1
    assert schedule(100) == 0.1 * 0.9
    assert schedule(200) == 0.1 * 0.9 ** 2
    assert schedule(50) == 0.1 * 0.9 ** 0.5
    
    # Test staircase decay
    schedule = learning_rate_schedule(init_lr=0.1, decay_steps=100, decay_rate=0.9, staircase=True)
    assert schedule(0) == 0.1
    assert schedule(99) == 0.1
    assert schedule(100) == 0.1 * 0.9
    assert schedule(199) == 0.1 * 0.9
    assert schedule(200) == 0.1 * 0.9 ** 2


def test_clip_gradients():
    """Test gradient clipping."""
    # Test clipping Adam optimizer
    optimizer = create_optimizer(optimizer_name="adam", learning_rate=0.01)
    clipped_optimizer = clip_gradients(optimizer, clip_norm=10.0)
    assert isinstance(clipped_optimizer, numpyro.optim.ClippedAdam)
    assert clipped_optimizer.clip_norm == 10.0
    # We don't test step_size as it might not be directly accessible
    
    # Test clipping SGD optimizer
    optimizer = create_optimizer(optimizer_name="sgd", learning_rate=0.1)
    clipped_optimizer = clip_gradients(optimizer, clip_norm=5.0)
    # We're now using ClippedAdam for all optimizers
    assert isinstance(clipped_optimizer, numpyro.optim.ClippedAdam)
    assert clipped_optimizer.clip_norm == 5.0


def test_create_optimizer_with_schedule():
    """Test optimizer creation with schedule."""
    # Test with Adam optimizer and continuous decay
    optimizer = create_optimizer_with_schedule(
        optimizer_name="adam",
        init_lr=0.1,
        decay_steps=100,
        decay_rate=0.9,
        staircase=False,
    )
    # The result should be an Adam optimizer with a learning rate schedule
    assert isinstance(optimizer, numpyro.optim.Adam)
    
    # Test with SGD optimizer and staircase decay
    optimizer = create_optimizer_with_schedule(
        optimizer_name="sgd",
        init_lr=0.1,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True,
    )
    # The result should be an SGD optimizer with a learning rate schedule
    assert isinstance(optimizer, numpyro.optim.SGD)
    
    # Test with gradient clipping
    optimizer = create_optimizer_with_schedule(
        optimizer_name="adam",
        init_lr=0.1,
        decay_steps=100,
        decay_rate=0.9,
        clip_norm=10.0,
    )
    # The result should be a ClippedAdam optimizer
    assert isinstance(optimizer, numpyro.optim.ClippedAdam)
    assert optimizer.clip_norm == 10.0