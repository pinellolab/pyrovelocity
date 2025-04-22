"""
Tests for the SVI inference components.
"""

import pyro
import pyro.distributions as dist
import pytest
import torch

# These will be needed when we implement the guide tests
# from pyrovelocity.models.modular.components.guides import (
#     AutoGuideFactory,
#     DeltaGuide,
#     NormalGuide,
# )
from pyrovelocity.models.modular.inference.config import create_inference_config
from pyrovelocity.models.modular.inference.svi import (
    TrainingState,
    create_optimizer,
    create_svi,
    extract_posterior_samples,
    run_svi_inference,
    svi_step,
)


# Define a simple model for testing
def simple_model(x=None):
    # Sample parameters from prior
    alpha = pyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = pyro.sample("beta", dist.LogNormal(0.0, 1.0))
    gamma = pyro.sample("gamma", dist.LogNormal(0.0, 1.0))

    # Generate data
    if x is not None:
        # Use plate to handle batched data
        with pyro.plate("data", len(x)):
            # Sample observations
            pyro.sample("obs", dist.Normal(alpha * x, beta), obs=x * gamma)

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


# Define a simple guide for testing
def simple_guide(x=None):  # x is unused but required for API compatibility
    # Variational parameters
    alpha_loc = pyro.param("alpha_loc", torch.tensor(0.0))
    alpha_scale = pyro.param("alpha_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    beta_loc = pyro.param("beta_loc", torch.tensor(0.0))
    beta_scale = pyro.param("beta_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    gamma_loc = pyro.param("gamma_loc", torch.tensor(0.0))
    gamma_scale = pyro.param("gamma_scale", torch.tensor(1.0), constraint=dist.constraints.positive)

    # Sample from variational distributions
    alpha = pyro.sample("alpha", dist.LogNormal(alpha_loc, alpha_scale))
    beta = pyro.sample("beta", dist.LogNormal(beta_loc, beta_scale))
    gamma = pyro.sample("gamma", dist.LogNormal(gamma_loc, gamma_scale))

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


class TestSVI:
    """Tests for SVI inference."""

    def test_create_optimizer(self):
        """Test creating an optimizer."""
        # Test creating Adam optimizer
        optimizer = create_optimizer("adam", learning_rate=0.01)
        assert optimizer is not None

        # Test creating SGD optimizer
        optimizer = create_optimizer("sgd", learning_rate=0.01)
        assert optimizer is not None

        # Test creating RMSprop optimizer
        optimizer = create_optimizer("rmsprop", learning_rate=0.01)
        assert optimizer is not None

        # Test invalid optimizer
        with pytest.raises(ValueError):
            create_optimizer("invalid", learning_rate=0.01)

    def test_create_svi(self):
        """Test creating an SVI object."""
        # Test creating SVI with string optimizer
        svi = create_svi(simple_model, simple_guide, "adam", learning_rate=0.01)
        assert svi is not None

        # Test creating SVI with optimizer object
        optimizer = create_optimizer("adam", learning_rate=0.01)
        svi = create_svi(simple_model, simple_guide, optimizer)
        assert svi is not None

    def test_svi_step(self):
        """Test performing an SVI step."""
        # Create SVI object
        svi = create_svi(simple_model, simple_guide, "adam", learning_rate=0.01)

        # Generate data
        x = torch.randn(10)

        # Perform SVI step
        loss = svi_step(svi, x)
        assert isinstance(loss, float)

    def test_extract_posterior_samples(self):
        """Test extracting posterior samples."""
        # Create SVI object
        svi = create_svi(simple_model, simple_guide, "adam", learning_rate=0.01)

        # Generate data
        x = torch.randn(10)

        # Perform SVI steps
        for _ in range(10):
            svi_step(svi, x)

        # Extract posterior samples
        # In Pyro, we need to use the guide directly to get parameters
        samples = extract_posterior_samples(simple_guide, {}, num_samples=100)
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 100

    def test_run_svi_inference(self):
        """Test running SVI inference."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="svi",
            num_epochs=10,
            num_samples=100,
            learning_rate=0.01,
            optimizer="adam",
        )

        # Run SVI inference
        state, samples = run_svi_inference(
            simple_model,
            simple_guide,
            args=(x,),
            config=config,
        )

        # Check results
        assert isinstance(state, TrainingState)
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 100

    def test_auto_guide_factory(self):
        """Test using AutoGuideFactory."""
        # Skip this test for now as it requires more work to integrate with Pyro's AutoGuide
        pytest.skip("AutoGuideFactory integration needs more work")

    def test_normal_guide(self):
        """Test using NormalGuide."""
        # Skip this test for now as it requires more work to integrate with Pyro's guide system
        pytest.skip("NormalGuide integration needs more work")

    def test_delta_guide(self):
        """Test using DeltaGuide."""
        # Skip this test for now as it requires more work to integrate with Pyro's guide system
        pytest.skip("DeltaGuide integration needs more work")
