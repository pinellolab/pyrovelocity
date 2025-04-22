"""
Tests for the posterior analysis components.
"""

import numpy as np
import pyro
import pyro.distributions as dist
import pytest
import torch
from anndata import AnnData

from pyrovelocity.models.modular.inference.config import (
    InferenceConfig,
    create_inference_config,
    validate_config,
)
from pyrovelocity.models.modular.inference.posterior import (
    analyze_posterior,
    compute_uncertainty,
    compute_velocity,
    create_inference_data,
    format_anndata_output,
    sample_posterior,
)
from pyrovelocity.models.modular.inference.unified import (
    InferenceState,
    create_inference_state,
    run_inference,
)


# Define a simple model for testing
def simple_model(x=None):
    # Sample parameters from prior
    alpha = pyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = pyro.sample("beta", dist.LogNormal(0.0, 1.0))
    gamma = pyro.sample("gamma", dist.LogNormal(0.0, 1.0))

    # Generate data
    if x is not None:
        u = pyro.sample("u", dist.Normal(alpha / beta, 0.1))
        s = pyro.sample("s", dist.Normal(alpha / gamma, 0.1))
        # Sample observations
        with pyro.plate("data", len(x)):
            pyro.sample("obs", dist.Normal(alpha * x, beta), obs=x * gamma)

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


# Define a simple guide for testing
def simple_guide(x=None):  # x is unused but required for API compatibility
    # Variational parameters
    alpha_loc = pyro.param("alpha_loc", torch.tensor(0.0))
    alpha_scale = pyro.param(
        "alpha_scale", torch.tensor(1.0), constraint=dist.constraints.positive
    )
    beta_loc = pyro.param("beta_loc", torch.tensor(0.0))
    beta_scale = pyro.param(
        "beta_scale", torch.tensor(1.0), constraint=dist.constraints.positive
    )
    gamma_loc = pyro.param("gamma_loc", torch.tensor(0.0))
    gamma_scale = pyro.param(
        "gamma_scale", torch.tensor(1.0), constraint=dist.constraints.positive
    )

    # Sample from variational distributions
    alpha = pyro.sample("alpha", dist.LogNormal(alpha_loc, alpha_scale))
    beta = pyro.sample("beta", dist.LogNormal(beta_loc, beta_scale))
    gamma = pyro.sample("gamma", dist.LogNormal(gamma_loc, gamma_scale))

    # Sample u and s
    u = pyro.sample("u", dist.Normal(alpha / beta, 0.1))
    s = pyro.sample("s", dist.Normal(alpha / gamma, 0.1))

    return {"alpha": alpha, "beta": beta, "gamma": gamma, "u": u, "s": s}


class TestPosterior:
    """Tests for posterior analysis."""

    def test_sample_posterior(self):
        """Test sampling from the posterior."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="svi",
            num_epochs=10,
            num_samples=10,
            learning_rate=0.01,
            optimizer="adam",
        )

        # Run inference
        state = run_inference(
            simple_model,
            simple_guide,
            args=(x,),
            config=config,
        )

        # Sample from posterior
        samples = sample_posterior(simple_model, state, num_samples=5)
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 5

    def test_compute_velocity(self):
        """Test computing RNA velocity."""
        # Generate posterior samples
        alpha = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        beta = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        gamma = torch.tensor([[0.3, 0.4], [0.5, 0.6]])
        u = torch.tensor([2.0, 3.0])
        s = torch.tensor([3.0, 4.0])
        posterior_samples = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "u": u,
            "s": s,
        }

        # Compute velocity
        velocity_results = compute_velocity(simple_model, posterior_samples)
        assert isinstance(velocity_results, dict)
        assert "velocity" in velocity_results
        assert "alpha" in velocity_results
        assert "beta" in velocity_results
        assert "gamma" in velocity_results
        assert "u_ss" in velocity_results
        assert "s_ss" in velocity_results
        assert velocity_results["velocity"].shape == (2, 2)

        # Compute velocity with mean
        velocity_results = compute_velocity(
            simple_model, posterior_samples, use_mean=True
        )
        assert isinstance(velocity_results, dict)
        assert "velocity" in velocity_results
        assert "alpha" in velocity_results
        assert "beta" in velocity_results
        assert "gamma" in velocity_results
        assert "u_ss" in velocity_results
        assert "s_ss" in velocity_results
        assert velocity_results["velocity"].shape == (2,)

    def test_compute_uncertainty(self):
        """Test computing uncertainty in RNA velocity."""
        # Generate velocity samples
        velocity = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Compute uncertainty with std method
        uncertainty = compute_uncertainty(velocity, method="std")
        assert isinstance(uncertainty, torch.Tensor)
        assert uncertainty.shape == (2,)

        # Compute uncertainty with quantile method
        uncertainty = compute_uncertainty(velocity, method="quantile")
        assert isinstance(uncertainty, torch.Tensor)
        assert uncertainty.shape == (2,)

        # Test invalid method
        with pytest.raises(ValueError):
            compute_uncertainty(velocity, method="invalid")

    def test_analyze_posterior(self):
        """Test analyzing posterior samples."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="svi",
            num_epochs=10,
            num_samples=10,
            learning_rate=0.01,
            optimizer="adam",
        )

        # Run inference
        state = run_inference(
            simple_model,
            simple_guide,
            args=(x,),
            config=config,
        )

        # Analyze posterior
        results = analyze_posterior(
            state,
            simple_model,
            num_samples=5,
            compute_velocity_flag=True,
            compute_uncertainty_flag=True,
        )
        assert isinstance(results, dict)
        assert "posterior_samples" in results
        assert "velocity" in results
        assert "uncertainty" in results
        assert results["posterior_samples"]["alpha"].shape[0] == 5

    def test_create_inference_data(self):
        """Test creating ArviZ InferenceData object."""
        # Generate posterior samples
        alpha = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        beta = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        gamma = torch.tensor([[0.3, 0.4], [0.5, 0.6]])
        posterior_samples = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }

        # Generate observed data
        u = torch.tensor([2.0, 3.0])
        s = torch.tensor([3.0, 4.0])
        observed_data = {
            "u": u,
            "s": s,
        }

        # Create inference data
        inference_data = create_inference_data(posterior_samples, observed_data)
        assert isinstance(inference_data, dict) or hasattr(
            inference_data, "posterior"
        )

    def test_format_anndata_output(self):
        """Test formatting results into AnnData object."""
        # Create AnnData object
        adata = AnnData(
            X=np.random.rand(10, 5),
            layers={
                "unspliced": np.random.rand(10, 5),
                "spliced": np.random.rand(10, 5),
            },
        )

        # Generate results
        # Create data with correct dimensions for AnnData (n_obs x n_vars)
        alpha = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
        )
        beta = torch.tensor(
            [[0.5, 0.6, 0.7, 0.8, 0.9], [1.0, 1.1, 1.2, 1.3, 1.4]]
        )
        gamma = torch.tensor(
            [[0.3, 0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1, 1.2]]
        )

        # Create velocity matrix with shape (n_obs, n_vars) = (10, 5)
        velocity = np.random.rand(10, 5)
        u_ss = np.random.rand(10, 5)
        s_ss = np.random.rand(10, 5)
        uncertainty = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        results = {
            "posterior_samples": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            },
            "alpha": alpha.mean(dim=0),
            "beta": beta.mean(dim=0),
            "gamma": gamma.mean(dim=0),
            "u_ss": u_ss,
            "s_ss": s_ss,
            "velocity": velocity,
            "uncertainty": uncertainty,
        }

        # Format results into AnnData object
        adata_out = format_anndata_output(adata, results)
        assert isinstance(adata_out, AnnData)
        assert "pyrovelocity_alpha" in adata_out.var
        assert "pyrovelocity_beta" in adata_out.var
        assert "pyrovelocity_gamma" in adata_out.var
        assert "pyrovelocity_uncertainty" in adata_out.var
        assert "pyrovelocity_velocity" in adata_out.layers
        assert "pyrovelocity_u_ss" in adata_out.layers
        assert "pyrovelocity_s_ss" in adata_out.layers
