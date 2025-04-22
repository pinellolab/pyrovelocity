"""
Tests for model comparison utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from pyrovelocity.models.jax.comparison.comparison import (
    compare_models,
    compute_log_likelihood,
    compute_loo,
    compute_waic,
)
from pyrovelocity.models.jax.core.state import InferenceState


@pytest.fixture
def test_model():
    """Create a simple test model."""

    def model(x=None, y=None):
        # Sample parameters
        alpha = numpyro.sample("alpha", dist.Normal(0, 1))
        beta = numpyro.sample("beta", dist.Normal(0, 1))

        # Compute mean
        mean = alpha + beta * x

        # Sample observations
        numpyro.sample("y", dist.Normal(mean, 1.0), obs=y)

        return {"mean": mean}

    return model


@pytest.fixture
def test_data():
    """Create test data."""
    # Set random seed for reproducibility
    np.random.seed(0)

    # Generate synthetic data
    x = jnp.array(np.random.normal(0, 1, 10))
    y = jnp.array(2.0 + 3.0 * x + np.random.normal(0, 1, 10))

    return x, y


@pytest.fixture
def test_inference_state():
    """Create a test inference state."""
    # Set random seed for reproducibility
    np.random.seed(0)

    # Create posterior samples
    posterior_samples = {
        "alpha": jnp.array(np.random.normal(2.0, 0.1, 100)),
        "beta": jnp.array(np.random.normal(3.0, 0.1, 100)),
    }

    # Create inference state
    inference_state = InferenceState(posterior_samples=posterior_samples)

    return inference_state


def test_compute_log_likelihood(test_model, test_data, test_inference_state):
    """Test computing log likelihood."""
    # Get test data and model
    x, y = test_data
    model = test_model

    # Get posterior samples
    posterior_samples = test_inference_state.posterior_samples

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Compute log likelihood
    log_likelihoods = compute_log_likelihood(
        model=model,
        posterior_samples=posterior_samples,
        args=(),
        kwargs={"x": x, "y": y},
        num_samples=10,
        key=key,
    )

    # Check shape
    assert log_likelihoods.shape == (10,)

    # Check values are finite
    assert jnp.all(jnp.isfinite(log_likelihoods))


def test_compute_waic(test_model, test_data, test_inference_state):
    """Test computing WAIC."""
    # Get test data and model
    x, y = test_data
    model = test_model

    # Get posterior samples
    posterior_samples = test_inference_state.posterior_samples

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Compute log likelihood
    log_likelihoods = compute_log_likelihood(
        model=model,
        posterior_samples=posterior_samples,
        args=(),
        kwargs={"x": x, "y": y},
        num_samples=10,
        key=key,
    )

    # Compute WAIC
    waic, waic_se, p_waic = compute_waic(log_likelihoods)

    # Check values are finite
    assert jnp.isfinite(waic)
    assert jnp.isfinite(waic_se)
    assert jnp.isfinite(p_waic)


def test_compute_loo(test_model, test_data, test_inference_state):
    """Test computing LOO."""
    # Get test data and model
    x, y = test_data
    model = test_model

    # Get posterior samples
    posterior_samples = test_inference_state.posterior_samples

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Compute log likelihood
    log_likelihoods = compute_log_likelihood(
        model=model,
        posterior_samples=posterior_samples,
        args=(),
        kwargs={"x": x, "y": y},
        num_samples=10,
        key=key,
    )

    # Compute LOO
    loo, loo_se, p_loo = compute_loo(log_likelihoods)

    # Check values are finite
    assert jnp.isfinite(loo)
    assert jnp.isfinite(loo_se)
    assert jnp.isfinite(p_loo)


def test_compare_models(test_model, test_data, test_inference_state):
    """Test comparing models."""
    # Get test data and model
    x, y = test_data
    model = test_model

    # Create a second model
    def model2(x=None, y=None):
        # Sample parameters
        alpha = numpyro.sample("alpha", dist.Normal(0, 1))

        # Compute mean
        mean = alpha + 0.0 * x

        # Sample observations
        numpyro.sample("y", dist.Normal(mean, 1.0), obs=y)

        return {"mean": mean}

    # Create a second inference state
    posterior_samples2 = {
        "alpha": jnp.array(np.random.normal(2.0, 0.1, 100)),
    }
    inference_state2 = InferenceState(posterior_samples=posterior_samples2)

    # Create models dictionary
    models = {
        "model1": (model, test_inference_state),
        "model2": (model2, inference_state2),
    }

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Compare models
    results = compare_models(
        models=models,
        args=(),
        kwargs={"x": x, "y": y},
        num_samples=10,
        key=key,
    )

    # Check results
    assert "model1" in results
    assert "model2" in results
    assert "log_likelihood" in results["model1"]
    assert "waic" in results["model1"]
    assert "loo" in results["model1"]
    assert "weight" in results["model1"]
