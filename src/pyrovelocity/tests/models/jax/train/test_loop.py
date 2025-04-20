"""
Tests for PyroVelocity JAX/NumPyro training loop utilities.

This module contains tests for the training loop utilities, including:

- test_train_epoch: Test training for one epoch
- test_evaluate_model: Test model evaluation
- test_train_with_early_stopping: Test training with early stopping
- test_train_model: Test model training
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide

from pyrovelocity.models.jax.train.loop import (
    train_epoch,
    evaluate_model,
    train_with_early_stopping,
    train_model,
)
from pyrovelocity.models.jax.core.state import TrainingState


# Define a simple model for testing
def simple_model(x=None, y=None):
    """Simple linear regression model for testing."""
    # Sample parameters
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    
    # Compute mean
    mean = w * x + b
    
    # Sample observations
    with numpyro.plate("data", x.shape[0]):
        numpyro.sample("y", dist.Normal(mean, 0.1), obs=y)
    
    return mean


@pytest.fixture
def svi_fixture():
    """Fixture for SVI object and initial state."""
    # Set random seed for reproducibility
    rng_key = jax.random.PRNGKey(0)
    
    # Generate synthetic data
    n_data = 100
    true_w = 2.0
    true_b = 1.0
    x = jnp.linspace(-1, 1, n_data)
    y = true_w * x + true_b + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n_data,))
    
    # Create data dictionary
    data = {"x": x, "y": y}
    
    # Create guide
    guide = autoguide.AutoNormal(simple_model)
    
    # Create optimizer
    optimizer = numpyro.optim.Adam(step_size=0.01)
    
    # Create SVI object
    svi = SVI(simple_model, guide, optimizer, loss=Trace_ELBO())
    
    # Initialize parameters
    params = svi.init(rng_key, **data)
    
    # Create initial state
    initial_state = TrainingState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
        loss_history=[],
        best_params=None,
        best_loss=None,
        key=rng_key,
    )
    
    return svi, initial_state, data


def test_train_epoch(svi_fixture):
    """Test training for one epoch."""
    svi, initial_state, data = svi_fixture
    
    # Train for one epoch
    updated_state = train_epoch(svi, initial_state, data)
    
    # Check that state was updated
    assert updated_state.step == initial_state.step + 1
    assert len(updated_state.loss_history) == len(initial_state.loss_history) + 1
    assert updated_state.params != initial_state.params
    assert updated_state.opt_state != initial_state.opt_state


def test_train_epoch_with_batching(svi_fixture):
    """Test training for one epoch with batching."""
    svi, initial_state, data = svi_fixture
    
    # Train for one epoch with batching
    batch_size = 10
    updated_state = train_epoch(svi, initial_state, data, batch_size=batch_size)
    
    # Check that state was updated
    assert updated_state.step == initial_state.step + 1
    assert len(updated_state.loss_history) == len(initial_state.loss_history) + 1
    assert updated_state.params != initial_state.params
    assert updated_state.opt_state != initial_state.opt_state


def test_evaluate_model(svi_fixture):
    """Test model evaluation."""
    svi, initial_state, data = svi_fixture
    
    # Evaluate model
    loss = evaluate_model(svi, initial_state, data)
    
    # Check that loss is a float
    assert isinstance(loss, float)
    
    # Check that loss is positive (negative ELBO)
    assert loss > 0


def test_train_with_early_stopping(svi_fixture):
    """Test training with early stopping."""
    svi, initial_state, data = svi_fixture
    
    # Split data into train and validation sets
    n_data = len(data["x"])
    n_train = int(0.8 * n_data)
    
    train_data = {
        "x": data["x"][:n_train],
        "y": data["y"][:n_train],
    }
    
    val_data = {
        "x": data["x"][n_train:],
        "y": data["y"][n_train:],
    }
    
    # Train with early stopping
    num_epochs = 10
    patience = 3
    final_state = train_with_early_stopping(
        svi,
        initial_state,
        train_data,
        val_data,
        num_epochs=num_epochs,
        patience=patience,
        verbose=False,
    )
    
    # Check that state was updated
    assert final_state.step > initial_state.step
    assert len(final_state.loss_history) > len(initial_state.loss_history)
    assert final_state.params != initial_state.params
    assert final_state.opt_state != initial_state.opt_state
    
    # Check that best parameters were saved
    assert final_state.best_params is not None
    assert final_state.best_loss is not None


def test_train_model_with_early_stopping(svi_fixture):
    """Test model training with early stopping."""
    svi, initial_state, data = svi_fixture
    
    # Train model with early stopping
    num_epochs = 10
    final_state = train_model(
        svi,
        initial_state,
        data,
        num_epochs=num_epochs,
        early_stopping=True,
        early_stopping_patience=3,
        verbose=False,
    )
    
    # Check that state was updated
    assert final_state.step > initial_state.step
    assert len(final_state.loss_history) > len(initial_state.loss_history)
    assert final_state.params != initial_state.params
    assert final_state.opt_state != initial_state.opt_state
    
    # Check that best parameters were saved
    assert final_state.best_params is not None
    assert final_state.best_loss is not None


def test_train_model_without_early_stopping(svi_fixture):
    """Test model training without early stopping."""
    svi, initial_state, data = svi_fixture
    
    # Train model without early stopping
    num_epochs = 5
    final_state = train_model(
        svi,
        initial_state,
        data,
        num_epochs=num_epochs,
        early_stopping=False,
        verbose=False,
    )
    
    # Check that state was updated
    assert final_state.step == initial_state.step + num_epochs
    assert len(final_state.loss_history) == len(initial_state.loss_history) + num_epochs
    assert final_state.params != initial_state.params
    assert final_state.opt_state != initial_state.opt_state