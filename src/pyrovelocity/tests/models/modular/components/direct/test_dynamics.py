"""Tests for Protocol-First dynamics model implementations."""

import pytest
import torch

from pyrovelocity.models.modular.components.direct.dynamics import StandardDynamicsModelDirect
from pyrovelocity.models.modular.interfaces import DynamicsModel
from pyrovelocity.models.modular.registry import DynamicsModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_dynamics_models():
    """Register dynamics models for testing."""
    # Save original registry state
    original_registry = dict(DynamicsModelRegistry._registry)
    
    # Clear registry and register test components
    DynamicsModelRegistry.clear()
    DynamicsModelRegistry._registry["standard_direct"] = StandardDynamicsModelDirect
    
    yield
    
    # Restore original registry state
    DynamicsModelRegistry._registry = original_registry


def test_standard_dynamics_model_direct_registration():
    """Test that StandardDynamicsModelDirect is properly registered."""
    model_class = DynamicsModelRegistry.get("standard_direct")
    assert model_class == StandardDynamicsModelDirect
    assert model_class.name == "standard_direct"
    assert "standard_direct" in DynamicsModelRegistry.list_available()


def test_standard_dynamics_model_direct_initialization():
    """Test initialization of StandardDynamicsModelDirect."""
    model = StandardDynamicsModelDirect()
    assert model.name == "dynamics_model_direct"
    assert model.shared_time is True
    assert model.t_scale_on is False
    assert model.cell_specific_kinetics is None
    assert model.kinetics_num is None
    
    model = StandardDynamicsModelDirect(
        name="custom_name",
        shared_time=False,
        t_scale_on=True,
        cell_specific_kinetics="test",
        kinetics_num=5,
    )
    assert model.name == "custom_name"
    assert model.shared_time is False
    assert model.t_scale_on is True
    assert model.cell_specific_kinetics == "test"
    assert model.kinetics_num == 5


def test_standard_dynamics_model_direct_protocol():
    """Test that StandardDynamicsModelDirect implements the DynamicsModel Protocol."""
    model = StandardDynamicsModelDirect()
    assert isinstance(model, DynamicsModel)


def test_standard_dynamics_model_direct_forward():
    """Test forward method of StandardDynamicsModelDirect."""
    model = StandardDynamicsModelDirect()
    
    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.rand(batch_size, n_genes)
    s_obs = torch.rand(batch_size, n_genes)
    alpha = torch.rand(n_genes)
    beta = torch.rand(n_genes)
    gamma = torch.rand(n_genes)
    t = torch.rand(batch_size, 1)
    
    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "t": t,
    }
    
    # Call forward method
    result = model.forward(context)
    
    # Check that the context is updated with expected counts
    assert "u_expected" in result
    assert "s_expected" in result
    assert result["u_expected"].shape == (batch_size, n_genes)
    assert result["s_expected"].shape == (batch_size, n_genes)


def test_standard_dynamics_model_direct_steady_state():
    """Test steady_state method of StandardDynamicsModelDirect."""
    model = StandardDynamicsModelDirect()
    
    # Create test data
    n_genes = 4
    alpha = torch.rand(n_genes)
    beta = torch.rand(n_genes)
    gamma = torch.rand(n_genes)
    
    # Call steady_state method
    u_ss, s_ss = model.steady_state(alpha, beta, gamma)
    
    # Check that the steady state is computed correctly
    assert u_ss.shape == (n_genes,)
    assert s_ss.shape == (n_genes,)
    assert torch.allclose(u_ss, alpha / beta)
    assert torch.allclose(s_ss, alpha / gamma)


def test_standard_dynamics_model_direct_predict_future_states():
    """Test predict_future_states method of StandardDynamicsModelDirect."""
    model = StandardDynamicsModelDirect()
    
    # Create test data
    n_genes = 4
    u_current = torch.rand(n_genes)
    s_current = torch.rand(n_genes)
    alpha = torch.rand(n_genes)
    beta = torch.rand(n_genes)
    gamma = torch.rand(n_genes)
    time_delta = 0.5
    
    # Call predict_future_states method
    u_future, s_future = model.predict_future_states(
        (u_current, s_current), time_delta, alpha, beta, gamma
    )
    
    # Check that the future states are computed
    assert u_future.shape == (n_genes,)
    assert s_future.shape == (n_genes,)
    
    # Check that the future states are different from the current states
    assert not torch.allclose(u_future, u_current)
    assert not torch.allclose(s_future, s_current)


def test_standard_dynamics_model_direct_simulate():
    """Test simulate method of StandardDynamicsModelDirect."""
    model = StandardDynamicsModelDirect()
    
    # Create test data
    n_genes = 4
    u0 = torch.rand(n_genes)
    s0 = torch.rand(n_genes)
    alpha = torch.rand(n_genes)
    beta = torch.rand(n_genes)
    gamma = torch.rand(n_genes)
    t_max = 10.0
    n_steps = 5
    
    # Call simulate method
    t, u, s = model.simulate(u0, s0, alpha, beta, gamma, t_max=t_max, n_steps=n_steps)
    
    # Check that the simulation results are computed
    assert t.shape == (n_steps,)
    assert u.shape == (n_steps, n_genes)
    assert s.shape == (n_steps, n_genes)
    
    # Check that the first time point is 0
    assert t[0] == 0.0
    
    # Check that the last time point is t_max
    assert t[-1] == t_max
    
    # Check that the initial states match
    assert torch.allclose(u[0], u0)
    assert torch.allclose(s[0], s0)
