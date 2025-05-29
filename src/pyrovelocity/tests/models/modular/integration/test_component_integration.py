"""
Component integration tests for the PyroVelocity modular implementation.

This module tests the interactions between different components in the
modular architecture, ensuring they work correctly together.
"""

import numpy as np
import pyro
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyrovelocity.models.modular.components.dynamics import (
    LegacyDynamicsModel,
    StandardDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    LegacyAutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    LegacyLikelihoodModel,
    PoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.modular.components.priors import LogNormalPriorModel
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel
from pyrovelocity.models.modular.registry import register_standard_components


@pytest.fixture
def simple_data():
    """Create a simple dataset for testing."""
    # Create a small dataset with random values
    n_cells = 20
    n_genes = 5

    u_obs = torch.poisson(torch.rand(n_cells, n_genes) * 5).float()
    s_obs = torch.poisson(torch.rand(n_cells, n_genes) * 5).float()

    # Create a DataLoader for batch processing
    dataset = TensorDataset(u_obs, s_obs)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "dataloader": dataloader,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture(scope="module", autouse=True)
def setup_registries():
    """Register standard components in registries before running tests."""
    register_standard_components()
    return


class TestComponentIntegration:
    """Tests for component interactions."""

    def test_dynamics_prior_integration(self):
        """Test that dynamics model and prior model work together."""
        # Create component models
        dynamics_model = StandardDynamicsModel()
        prior_model = LogNormalPriorModel()

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Sample parameters from prior
        n_genes = 5
        prior_params = prior_model.sample_parameters(n_genes=n_genes)

        # Verify that prior produces expected parameters
        assert "alpha" in prior_params
        assert "beta" in prior_params
        assert "gamma" in prior_params
        assert "u_scale" in prior_params
        assert "s_scale" in prior_params

        # Check shape of prior samples
        assert prior_params["alpha"].shape == (n_genes,)
        assert prior_params["beta"].shape == (n_genes,)
        assert prior_params["gamma"].shape == (n_genes,)

        # Use parameters with dynamics model
        u0 = torch.zeros(n_genes)
        s0 = torch.zeros(n_genes)

        # Simulate dynamics
        times, u, s = dynamics_model.simulate(
            u0=u0,
            s0=s0,
            alpha=prior_params["alpha"],
            beta=prior_params["beta"],
            gamma=prior_params["gamma"],
            t_max=10.0,
            n_steps=100,
        )

        # Verify simulation results
        assert times.shape == (100,)
        assert u.shape == (100, n_genes)
        assert s.shape == (100, n_genes)

        # Compute steady state
        u_ss, s_ss = dynamics_model.steady_state(
            prior_params["alpha"],
            prior_params["beta"],
            prior_params["gamma"],
        )

        # Verify steady state shapes
        assert u_ss.shape == (n_genes,)
        assert s_ss.shape == (n_genes,)

    def test_dynamics_likelihood_integration(self, simple_data):
        """Test that dynamics model and likelihood model work together."""
        # Create component models
        dynamics_model = StandardDynamicsModel()
        likelihood_model = PoissonLikelihoodModel()

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create parameters
        n_genes = simple_data["n_genes"]
        alpha = torch.rand(n_genes) * 5 + 1  # [1, 6]
        beta = torch.rand(n_genes) * 2 + 0.5  # [0.5, 2.5]
        gamma = torch.rand(n_genes) * 1 + 0.2  # [0.2, 1.2]

        # Compute steady state
        u_ss, s_ss = dynamics_model.steady_state(alpha, beta, gamma)

        # Create expected values using steady state - use a small batch size to match test
        batch_size = 10  # Use a smaller batch size to test
        u_expected = u_ss.unsqueeze(0).expand(batch_size, -1)
        s_expected = s_ss.unsqueeze(0).expand(batch_size, -1)

        # Create subset of observations for testing
        u_obs = simple_data["u_obs"][:batch_size]
        s_obs = simple_data["s_obs"][:batch_size]

        # Use log_prob directly instead of the __call__ interface that uses plates
        # Convert numpy arrays to torch tensors to satisfy jaxtyping constraints
        u_log_prob = likelihood_model.log_prob(
            observations=u_obs,  # Keep as torch tensor
            predictions=u_expected,  # Keep as torch tensor
        )
        s_log_prob = likelihood_model.log_prob(
            observations=s_obs,  # Keep as torch tensor
            predictions=s_expected,  # Keep as torch tensor
        )

        # Verify that log probabilities have the right shape
        assert u_log_prob.shape == (batch_size,)
        assert s_log_prob.shape == (batch_size,)

    def test_full_model_integration(self, simple_data):
        """Test that all components work together in the full model."""
        # Create components
        dynamics_model = StandardDynamicsModel()
        prior_model = LogNormalPriorModel()
        likelihood_model = PoissonLikelihoodModel()
        guide_model = AutoGuideFactory(guide_type="AutoNormal")

        # Create the full model
        model = PyroVelocityModel(
            dynamics_model=dynamics_model,
            prior_model=prior_model,
            likelihood_model=likelihood_model,
            guide_model=guide_model,
        )

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create a simple dataset with fixed dimensions
        n_cells = 5
        n_genes = 5
        u_batch = torch.rand(n_cells, n_genes) * 5
        s_batch = torch.rand(n_cells, n_genes) * 5

        # Create parameters for the model
        alpha = torch.rand(n_genes) * 5 + 1  # [1, 6]
        beta = torch.rand(n_genes) * 2 + 0.5  # [0.5, 2.5]
        gamma = torch.rand(n_genes) * 1 + 0.2  # [0.2, 1.2]

        # Compute steady state values
        u_ss, s_ss = dynamics_model.steady_state(alpha, beta, gamma)

        # Expand to match batch size
        u_expected = u_ss.unsqueeze(0).expand(n_cells, -1)
        s_expected = s_ss.unsqueeze(0).expand(n_cells, -1)

        # Create a context dictionary with the data
        context = {
            "u_obs": u_batch,
            "s_obs": s_batch,
            "u_expected": u_expected,
            "s_expected": s_expected,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }

        # Call the model directly with the context
        result = model.forward(**context)

        # Verify that the model produced expected outputs
        assert "u_dist" in result
        assert "s_dist" in result

    def test_different_guides(self, simple_data):
        """Test that different guide implementations work with the same model."""
        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create common components
        dynamics_model = StandardDynamicsModel()
        prior_model = LogNormalPriorModel()
        likelihood_model = PoissonLikelihoodModel()

        # Create a simple dataset with fixed dimensions
        n_cells = 5
        n_genes = 5
        u_batch = torch.rand(n_cells, n_genes) * 5
        s_batch = torch.rand(n_cells, n_genes) * 5

        # Create parameters for the model
        alpha = torch.rand(n_genes) * 5 + 1  # [1, 6]
        beta = torch.rand(n_genes) * 2 + 0.5  # [0.5, 2.5]
        gamma = torch.rand(n_genes) * 1 + 0.2  # [0.2, 1.2]

        # Compute steady state values
        u_ss, s_ss = dynamics_model.steady_state(alpha, beta, gamma)

        # Expand to match batch size
        u_expected = u_ss.unsqueeze(0).expand(n_cells, -1)
        s_expected = s_ss.unsqueeze(0).expand(n_cells, -1)

        # Create a context dictionary with the data
        context = {
            "u_obs": u_batch,
            "s_obs": s_batch,
            "u_expected": u_expected,
            "s_expected": s_expected,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }

        # Create different guide types to test
        guide_types = [
            AutoGuideFactory(guide_type="AutoNormal"),
            LegacyAutoGuideFactory(add_offset=True),
        ]

        # Test each guide
        for guide_model in guide_types:
            # Create the full model
            model = PyroVelocityModel(
                dynamics_model=dynamics_model,
                prior_model=prior_model,
                likelihood_model=likelihood_model,
                guide_model=guide_model,
            )

            # Call the model directly with the context
            result = model.forward(**context)

            # Verify that the model produced expected outputs
            assert "u_dist" in result
            assert "s_dist" in result
