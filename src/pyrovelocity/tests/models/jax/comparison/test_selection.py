"""
Tests for model selection utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from pyrovelocity.models.jax.comparison.selection import (
    compute_predictive_performance,
    cross_validate,
    select_best_model,
)
from pyrovelocity.models.jax.core.state import InferenceState


@pytest.fixture
def test_models():
    """Create test models."""

    def model1(x=None, y=None):
        # Sample parameters
        alpha = numpyro.sample("alpha", dist.Normal(0, 1))
        beta = numpyro.sample("beta", dist.Normal(0, 1))

        # Compute mean
        mean = alpha + beta * x

        # Sample observations
        numpyro.sample("y", dist.Normal(mean, 1.0), obs=y)

        return {"mean": mean}

    def model2(x=None, y=None):
        # Sample parameters
        alpha = numpyro.sample("alpha", dist.Normal(0, 1))

        # Compute mean (no slope)
        mean = alpha + 0.0 * x

        # Sample observations
        numpyro.sample("y", dist.Normal(mean, 1.0), obs=y)

        return {"mean": mean}

    return model1, model2


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
def test_inference_states():
    """Create test inference states."""
    # Set random seed for reproducibility
    np.random.seed(0)

    # Create posterior samples for model1
    posterior_samples1 = {
        "alpha": jnp.array(np.random.normal(2.0, 0.1, 100)),
        "beta": jnp.array(np.random.normal(3.0, 0.1, 100)),
    }

    # Create posterior samples for model2
    posterior_samples2 = {
        "alpha": jnp.array(np.random.normal(2.0, 0.1, 100)),
    }

    # Create inference states
    inference_state1 = InferenceState(posterior_samples=posterior_samples1)
    inference_state2 = InferenceState(posterior_samples=posterior_samples2)

    return inference_state1, inference_state2


def test_select_best_model(test_models, test_data, test_inference_states):
    """Test selecting the best model."""
    # Get test data and models
    x, y = test_data
    model1, model2 = test_models
    inference_state1, inference_state2 = test_inference_states

    # Create models dictionary
    models = {
        "model1": (model1, inference_state1),
        "model2": (model2, inference_state2),
    }

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Select best model using WAIC
    best_model_name, comparison_results = select_best_model(
        models=models,
        args=(),
        kwargs={"x": x, "y": y},
        criterion="waic",
        num_samples=10,
        key=key,
    )

    # Check that best model is selected
    assert best_model_name in ["model1", "model2"]
    assert "model1" in comparison_results
    assert "model2" in comparison_results
    assert "waic" in comparison_results["model1"]
    assert "loo" in comparison_results["model1"]


def test_compute_predictive_performance(
    test_models, test_data, test_inference_states
):
    """Test computing predictive performance."""
    # Get test data and models
    x, y = test_data
    model1, _ = test_models
    inference_state1, _ = test_inference_states

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Compute predictive performance
    performance = compute_predictive_performance(
        model=model1,
        inference_state=inference_state1,
        test_args=(),
        test_kwargs={"x": x, "y": y},
        num_samples=10,
        key=key,
    )

    # Check that performance metrics are computed
    assert "log_likelihood" in performance
    assert "waic" in performance
    assert "loo" in performance
    assert "p_waic" in performance
    assert "p_loo" in performance


def test_cross_validate(test_models, test_data):
    """Test cross-validation."""
    # Get test data and models
    x, y = test_data
    model1, _ = test_models

    # Create data splits
    data_splits = [
        ((), {"x": x[:5], "y": y[:5]}, (), {"x": x[5:], "y": y[5:]}),
        ((), {"x": x[5:], "y": y[5:]}, (), {"x": x[:5], "y": y[:5]}),
    ]

    # Create a simple inference function
    def inference_fn(model, args, kwargs, key):
        # Create mock posterior samples
        posterior_samples = {
            "alpha": jnp.array(np.random.normal(2.0, 0.1, 100)),
            "beta": jnp.array(np.random.normal(3.0, 0.1, 100)),
        }
        return InferenceState(posterior_samples=posterior_samples)

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Perform cross-validation
    cv_results = cross_validate(
        model_fn=model1,
        data_splits=data_splits,
        inference_fn=inference_fn,
        num_samples=10,
        key=key,
    )

    # Check that cross-validation results are computed
    assert "log_likelihood" in cv_results
    assert "waic" in cv_results
    assert "loo" in cv_results
    assert "log_likelihood_mean" in cv_results
    assert "waic_mean" in cv_results
    assert "loo_mean" in cv_results
    assert "log_likelihood_std" in cv_results
    assert "waic_std" in cv_results
    assert "loo_std" in cv_results
