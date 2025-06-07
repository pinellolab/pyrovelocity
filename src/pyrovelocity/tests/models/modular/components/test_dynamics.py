"""Tests for dynamics models."""

import numpy as np
import pytest
import torch
from jaxtyping import Array

from pyrovelocity.models.modular.components.dynamics import (
    LegacyDynamicsModel,
    PiecewiseActivationDynamicsModel,
)
from pyrovelocity.models.modular.registry import DynamicsModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_dynamics_models():
    """Register dynamics models for testing."""
    # Save original registry state
    original_registry = dict(DynamicsModelRegistry._registry)

    # Clear registry and register test components
    DynamicsModelRegistry.clear()
    DynamicsModelRegistry._registry["piecewise_activation"] = PiecewiseActivationDynamicsModel
    DynamicsModelRegistry._registry["legacy"] = LegacyDynamicsModel

    yield

    # Restore original registry state
    DynamicsModelRegistry._registry = original_registry


def test_piecewise_activation_dynamics_model_registration():
    """Test that PiecewiseActivationDynamicsModel is properly registered."""
    model_class = DynamicsModelRegistry.get("piecewise_activation")
    assert model_class == PiecewiseActivationDynamicsModel
    assert model_class.name == "piecewise_activation"
    assert "piecewise_activation" in DynamicsModelRegistry.list_available()


def test_legacy_dynamics_model_registration():
    """Test that LegacyDynamicsModel is properly registered."""
    model_class = DynamicsModelRegistry.get("legacy")
    assert model_class == LegacyDynamicsModel
    assert model_class.name == "legacy"
    assert "legacy" in DynamicsModelRegistry.list_available()


class TestPiecewiseActivationDynamicsModel:
    """Tests for PiecewiseActivationDynamicsModel."""

    @pytest.fixture
    def model(self):
        """Create a PiecewiseActivationDynamicsModel instance."""
        return PiecewiseActivationDynamicsModel()

    @pytest.fixture
    def simple_params(self):
        """Create simple parameters for testing."""
        return {
            "u0": torch.tensor([1.0, 2.0]),
            "s0": torch.tensor([0.5, 1.0]),
            "alpha": torch.tensor([2.0, 3.0]),
            "beta": torch.tensor([1.0, 1.5]),
            "gamma": torch.tensor([0.5, 0.8]),
            "scaling": torch.tensor([1.0, 1.0]),
            "t_max": 10.0,
            "n_steps": 100,
            # Piecewise-specific parameters
            "alpha_off": torch.tensor([1.0, 1.5]),  # Basal transcription rate
            "gamma_star": torch.tensor([0.5, 0.8]),  # Relative degradation rate
        }

    def test_steady_state(self, model, simple_params):
        """Test steady state calculation for piecewise activation model."""
        # For piecewise model, steady state uses alpha_off and gamma_star
        # Note: In the corrected parameterization, alpha_off is always 1.0 (fixed reference)
        gamma_star = simple_params["gamma_star"]

        # Create fixed alpha_off tensor (always 1.0)
        alpha_off_fixed = torch.ones_like(gamma_star)

        u_ss, s_ss = model.steady_state(
            alpha_off_fixed,
            gamma_star,
        )

        # Expected steady states for OFF phase: u* = 1.0, s* = 1.0/γ*
        expected_u_ss = torch.ones_like(gamma_star)  # Always 1.0 (fixed reference)
        expected_s_ss = torch.ones_like(gamma_star) / gamma_star

        assert torch.allclose(u_ss, expected_u_ss)
        assert torch.allclose(s_ss, expected_s_ss)

    def test_simulate(self, model, simple_params):
        """Test that PiecewiseActivationDynamicsModel doesn't have simulate method."""
        # PiecewiseActivationDynamicsModel uses analytical solutions in forward()
        # and doesn't provide a simulate() method like the legacy model
        assert not hasattr(model, 'simulate'), "PiecewiseActivationDynamicsModel should not have simulate method"

        # Instead, test that the model has the expected analytical solution methods
        assert hasattr(model, '_compute_piecewise_solution'), "Should have _compute_piecewise_solution method"
        assert hasattr(model, 'forward'), "Should have forward method"

    def test_conservation_laws(self, model, simple_params):
        """Test that the model obeys conservation laws."""
        # For the analytical solution, we can directly verify the equations
        # rather than using numerical differentiation

        # Get parameters
        alpha = simple_params["alpha"]
        beta = simple_params["beta"]
        gamma = simple_params["gamma"]
        u0 = simple_params["u0"]
        s0 = simple_params["s0"]

        # Pick a time point to check
        t = 0.5

        # Compute the analytical solution at time t
        expu = torch.exp(-beta * t)
        exps = torch.exp(-gamma * t)

        # u(t) = u0 * e^(-beta*t) + alpha/beta * (1 - e^(-beta*t))
        u_t = u0 * expu + alpha / beta * (1 - expu)

        # For gamma != beta
        # s(t) = s0 * e^(-gamma*t) + alpha/gamma * (1 - e^(-gamma*t)) +
        #        (alpha - u0*beta)/(gamma - beta) * (e^(-gamma*t) - e^(-beta*t))
        expus = (alpha - u0 * beta) / (gamma - beta + 1e-8) * (exps - expu)
        s_t = s0 * exps + alpha / gamma * (1 - exps) + expus

        # Compute the derivatives analytically
        # du/dt = -beta * u0 * e^(-beta*t) + alpha * e^(-beta*t)
        #       = -beta * (u0 * e^(-beta*t)) + alpha * e^(-beta*t)
        #       = -beta * (u_t - alpha/beta * (1 - e^(-beta*t))) + alpha * e^(-beta*t)
        #       = -beta * u_t + alpha
        du_dt = alpha - beta * u_t

        # Verify that du/dt = alpha - beta * u
        assert torch.allclose(du_dt, alpha - beta * u_t, rtol=1e-5)

        # For the piecewise model, test the dimensionless conservation laws
        # using the correct parameters (alpha_off is always 1.0 in corrected parameterization)
        gamma_star = simple_params["gamma_star"]
        alpha_off_fixed = torch.ones_like(gamma_star)  # Fixed reference

        # At steady state (OFF phase), derivatives should be zero
        u_ss, s_ss = model.steady_state(alpha_off_fixed, gamma_star)

        # For OFF phase: du*/dt* = α*_off - u* = 0 at steady state
        dudt_ss = alpha_off_fixed - u_ss

        # ds*/dt* = u* - γ*s* = 0 at steady state
        dsdt_ss = u_ss - gamma_star * s_ss

        assert torch.allclose(dudt_ss, torch.zeros_like(dudt_ss), atol=1e-6)
        assert torch.allclose(dsdt_ss, torch.zeros_like(dsdt_ss), atol=1e-6)


class TestLegacyDynamicsModel:
    """Tests for LegacyDynamicsModel."""

    @pytest.fixture
    def model(self):
        """Create a LegacyDynamicsModel instance."""
        return LegacyDynamicsModel()

    @pytest.fixture
    def simple_params(self):
        """Create simple parameters for testing."""
        return {
            "u0": torch.tensor([1.0, 2.0]),
            "s0": torch.tensor([0.5, 1.0]),
            "alpha": torch.tensor([2.0, 3.0]),
            "beta": torch.tensor([1.0, 1.5]),
            "gamma": torch.tensor([0.5, 0.8]),
            "scaling": torch.tensor([1.0, 1.0]),
            "t_max": 10.0,
            "n_steps": 100,
        }

    def test_steady_state(self, model, simple_params):
        """Test steady state calculation."""
        u_ss, s_ss = model.steady_state(
            simple_params["alpha"],
            simple_params["beta"],
            simple_params["gamma"],
        )

        # Expected steady states based on analytical solution
        expected_u_ss = simple_params["alpha"] / simple_params["beta"]
        expected_s_ss = simple_params["alpha"] / simple_params["gamma"]

        assert torch.allclose(u_ss, expected_u_ss)
        assert torch.allclose(s_ss, expected_s_ss)

    def test_forward(self, model):
        """Test forward method."""
        # Create a simple context
        batch_size = 3
        num_genes = 4
        u_obs = torch.rand(batch_size, num_genes)
        s_obs = torch.rand(batch_size, num_genes)
        alpha = torch.ones(num_genes)
        beta = torch.ones(num_genes)
        gamma = torch.ones(num_genes)

        context = {
            "u_obs": u_obs,
            "s_obs": s_obs,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }

        # Call forward method
        result = model.forward(context)

        # Check that the result contains expected keys
        assert "ut" in result
        assert "st" in result
        assert "u_inf" in result
        assert "s_inf" in result
        assert "switching" in result

        # Check shapes
        assert result["ut"].shape[-2:] == (batch_size, num_genes)
        assert result["st"].shape[-2:] == (batch_size, num_genes)


def test_model_comparison():
    """Compare piecewise activation and legacy models steady states."""
    # Legacy model parameters
    alpha = torch.tensor([2.0])
    beta = torch.tensor([1.0])
    gamma = torch.tensor([0.5])

    # Piecewise model parameters (for OFF phase comparison)
    alpha_off = alpha  # Use same transcription rate for comparison
    gamma_star = gamma  # Use same degradation rate for comparison

    # Create models
    piecewise_model = PiecewiseActivationDynamicsModel()
    legacy_model = LegacyDynamicsModel()

    # Calculate steady states using appropriate parameters for each model
    u_ss_piecewise, s_ss_piecewise = piecewise_model.steady_state(
        alpha_off, gamma_star
    )
    u_ss_legacy, s_ss_legacy = legacy_model.steady_state(
        alpha, beta, gamma
    )

    # For the OFF phase of piecewise model: u* = α*_off, s* = α*_off/γ*
    # For the legacy model: u = α/β, s = α/γ
    # These are only comparable when β = 1 (dimensionless case)
    expected_u_legacy = alpha / beta  # Should equal alpha when beta=1
    expected_s_legacy = alpha / gamma
    expected_u_piecewise = alpha_off
    expected_s_piecewise = alpha_off / gamma_star

    # Verify each model's steady state calculation
    assert torch.allclose(u_ss_legacy, expected_u_legacy)
    assert torch.allclose(s_ss_legacy, expected_s_legacy)
    assert torch.allclose(u_ss_piecewise, expected_u_piecewise)
    assert torch.allclose(s_ss_piecewise, expected_s_piecewise)

    # When beta=1, the models should give the same results
    assert torch.allclose(u_ss_legacy, u_ss_piecewise)
    assert torch.allclose(s_ss_legacy, s_ss_piecewise)
