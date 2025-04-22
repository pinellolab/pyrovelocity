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
    StandardDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    DeltaGuide,
    NormalGuide,
)
from pyrovelocity.models.modular.components.likelihoods import (
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
        u_log_prob = likelihood_model.log_prob(
            observations=u_obs.numpy(),
            predictions=u_expected.numpy(),
        )
        s_log_prob = likelihood_model.log_prob(
            observations=s_obs.numpy(),
            predictions=s_expected.numpy(),
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
        observation_model = StandardObservationModel()
        guide_model = AutoGuideFactory(guide_type="AutoNormal")

        # Create the full model
        model = PyroVelocityModel(
            dynamics_model=dynamics_model,
            prior_model=prior_model,
            likelihood_model=likelihood_model,
            observation_model=observation_model,
            guide_model=guide_model,
        )

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Initialize the guide with the model
        # This is necessary before using it with SVI
        guide_model.create_guide(model)

        # Create a simple SVI training loop
        optimizer = pyro.optim.Adam({"lr": 0.01})
        elbo = pyro.infer.Trace_ELBO()
        # Use the guide function returned by the guide model
        guide_fn = guide_model.get_guide()
        svi = pyro.infer.SVI(
            model=model, guide=guide_fn, optim=optimizer, loss=elbo
        )

        # Train for a few steps
        n_steps = 5
        losses = []

        for step in range(n_steps):
            for batch in simple_data["dataloader"]:
                u_batch, s_batch = batch
                loss = svi.step(u_obs=u_batch, s_obs=s_batch)
                losses.append(loss)

        # Verify that training produced some losses
        assert len(losses) == n_steps * len(simple_data["dataloader"])

    def test_different_guides(self, simple_data):
        """Test that different guide implementations work with the same model."""
        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create common components
        dynamics_model = StandardDynamicsModel()
        prior_model = LogNormalPriorModel()
        likelihood_model = PoissonLikelihoodModel()
        observation_model = StandardObservationModel()

        # Create different guides
        guides = {
            "auto_normal": AutoGuideFactory(guide_type="AutoNormal"),
            "auto_delta": AutoGuideFactory(guide_type="AutoDelta"),
            "normal": NormalGuide(),
            "delta": DeltaGuide(),
        }

        # Test each guide
        for guide_name, guide_model in guides.items():
            # Create the full model
            model = PyroVelocityModel(
                dynamics_model=dynamics_model,
                prior_model=prior_model,
                likelihood_model=likelihood_model,
                observation_model=observation_model,
                guide_model=guide_model,
            )

            # Initialize the guide with the model
            guide_model.create_guide(model)
            guide_fn = guide_model.get_guide()

            # Create a simple SVI training loop
            optimizer = pyro.optim.Adam({"lr": 0.01})
            elbo = pyro.infer.Trace_ELBO()
            svi = pyro.infer.SVI(
                model=model, guide=guide_fn, optim=optimizer, loss=elbo
            )

            # Train for just one step to verify it works
            # Just testing that the guide can be used with the model
            for batch in simple_data["dataloader"]:
                u_batch, s_batch = batch
                loss = svi.step(u_obs=u_batch, s_obs=s_batch)
                # Just verify that loss is a number
                assert isinstance(loss, float)
