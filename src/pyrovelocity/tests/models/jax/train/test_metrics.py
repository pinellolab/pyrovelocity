"""
Tests for PyroVelocity JAX/NumPyro training metrics.

This module contains tests for the training metrics, including:

- test_compute_loss: Test loss computation
- test_compute_elbo: Test ELBO computation
- test_compute_predictive_log_likelihood: Test predictive log likelihood computation
- test_compute_metrics: Test metrics computation
- test_compute_validation_metrics: Test validation metrics computation
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide

from pyrovelocity.models.jax.train.metrics import (
    compute_loss,
    compute_elbo,
    compute_predictive_log_likelihood,
    compute_metrics,
    compute_validation_metrics,
)


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
def model_fixture():
    """Fixture for model, guide, parameters, and data."""
    # Set random seed for reproducibility
    rng_key = jax.random.PRNGKey(0)

    # Generate synthetic data
    n_data = 100
    true_w = 2.0
    true_b = 1.0
    x = jnp.linspace(-1, 1, n_data)
    y = (
        true_w * x
        + true_b
        + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n_data,))
    )

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

    # Create mock posterior samples for testing
    # This avoids the issues with numpyro.infer.Predictive
    num_samples = 10
    mock_posterior_samples = {
        "w": jnp.ones((num_samples,)) * 2.0
        + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (num_samples,)),
        "b": jnp.ones((num_samples,)) * 1.0
        + 0.1 * jax.random.normal(jax.random.PRNGKey(3), (num_samples,)),
    }

    return simple_model, guide, svi, params, data, mock_posterior_samples

    return simple_model, guide, svi, params, data, posterior_samples


def test_compute_loss(model_fixture):
    """Test loss computation."""
    model, guide, svi, params, data, _ = model_fixture

    # Compute loss
    loss = compute_loss(svi, params, **data)

    # Check that loss is a float
    assert isinstance(loss, float)

    # Check that loss is positive (negative ELBO)
    assert loss > 0


def test_compute_elbo(model_fixture):
    """Test ELBO computation."""
    model, guide, _, params, data, _ = model_fixture

    # Compute ELBO
    elbo = compute_elbo(model, guide, params, **data)

    # Check that ELBO is a float
    assert isinstance(elbo, float)

    # Check that ELBO is negative (since it's the negative of the loss)
    assert elbo < 0


def test_compute_predictive_log_likelihood(model_fixture):
    """Test predictive log likelihood computation."""
    model, _, _, _, data, posterior_samples = model_fixture

    # Compute predictive log likelihood
    pred_ll = compute_predictive_log_likelihood(
        model, posterior_samples, **data
    )

    # Check that predictive log likelihood is a JAX array
    assert isinstance(pred_ll, jnp.ndarray)

    # Check shape (should be a scalar or vector depending on the model)
    assert pred_ll.ndim <= 1


def test_compute_metrics(model_fixture):
    """Test metrics computation."""
    model, guide, _, params, data, posterior_samples = model_fixture

    # Compute metrics
    metrics = compute_metrics(model, guide, params, posterior_samples, **data)

    # Check that metrics is a dictionary
    assert isinstance(metrics, dict)

    # Check that required metrics are present
    assert "elbo" in metrics
    assert "predictive_log_likelihood" in metrics
    assert "kl_divergence" in metrics
    assert "n_samples" in metrics

    # Check that metrics have correct types
    assert isinstance(metrics["elbo"], float)
    assert isinstance(metrics["predictive_log_likelihood"], float)
    assert isinstance(metrics["kl_divergence"], float)
    assert isinstance(metrics["n_samples"], int)

    # Check that parameter statistics are present
    assert any(key.endswith("_mean") for key in metrics)
    assert any(key.endswith("_std") for key in metrics)


def test_compute_validation_metrics(model_fixture):
    """Test validation metrics computation."""
    model, guide, _, params, data, _ = model_fixture

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

    # Compute validation metrics
    metrics = compute_validation_metrics(
        model, guide, params, train_data, val_data
    )

    # Check that metrics is a dictionary
    assert isinstance(metrics, dict)

    # Check that required metrics are present
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "generalization_gap" in metrics
    assert "train_predictive_log_likelihood" in metrics
    assert "val_predictive_log_likelihood" in metrics
    assert "predictive_performance_gap" in metrics

    # Check that metrics have correct types
    assert isinstance(metrics["train_loss"], float)
    assert isinstance(metrics["val_loss"], float)
    assert isinstance(metrics["generalization_gap"], float)
    assert isinstance(metrics["train_predictive_log_likelihood"], float)
    assert isinstance(metrics["val_predictive_log_likelihood"], float)
    assert isinstance(metrics["predictive_performance_gap"], float)

    # Check that generalization gap is consistent with losses
    np.testing.assert_allclose(
        metrics["generalization_gap"],
        metrics["val_loss"] - metrics["train_loss"],
        rtol=1e-5,
    )

    # Check that predictive performance gap is consistent with predictive log likelihoods
    np.testing.assert_allclose(
        metrics["predictive_performance_gap"],
        metrics["train_predictive_log_likelihood"]
        - metrics["val_predictive_log_likelihood"],
        rtol=1e-5,
    )
