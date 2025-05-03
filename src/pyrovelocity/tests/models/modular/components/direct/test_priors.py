"""Tests for Protocol-First prior model implementations."""

import pytest
import pyro
import torch

from pyrovelocity.models.modular.components.direct.priors import LogNormalPriorModelDirect
from pyrovelocity.models.modular.interfaces import PriorModel
from pyrovelocity.models.modular.registry import PriorModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_prior_models():
    """Register prior models for testing."""
    # Save original registry state
    original_registry = dict(PriorModelRegistry._registry)
    
    # Clear registry and register test components
    PriorModelRegistry.clear()
    PriorModelRegistry._registry["lognormal_direct"] = LogNormalPriorModelDirect
    
    yield
    
    # Restore original registry state
    PriorModelRegistry._registry = original_registry


def test_lognormal_prior_model_direct_registration():
    """Test that LogNormalPriorModelDirect is properly registered."""
    model_class = PriorModelRegistry.get("lognormal_direct")
    assert model_class == LogNormalPriorModelDirect
    assert model_class.name == "lognormal_direct"
    assert "lognormal_direct" in PriorModelRegistry.list_available()


def test_lognormal_prior_model_direct_initialization():
    """Test initialization of LogNormalPriorModelDirect."""
    model = LogNormalPriorModelDirect()
    assert model.name == "lognormal_direct"
    assert model.scale_alpha == 1.0
    assert model.scale_beta == 0.25
    assert model.scale_gamma == 1.0
    assert model.scale_u == 0.1
    assert model.scale_s == 0.1
    assert model.scale_dt == 1.0
    
    model = LogNormalPriorModelDirect(
        name="custom_name",
        scale_alpha=2.0,
        scale_beta=0.5,
        scale_gamma=1.5,
        scale_u=0.2,
        scale_s=0.3,
        scale_dt=0.5,
    )
    assert model.name == "custom_name"
    assert model.scale_alpha == 2.0
    assert model.scale_beta == 0.5
    assert model.scale_gamma == 1.5
    assert model.scale_u == 0.2
    assert model.scale_s == 0.3
    assert model.scale_dt == 0.5


def test_lognormal_prior_model_direct_protocol():
    """Test that LogNormalPriorModelDirect implements the PriorModel Protocol."""
    model = LogNormalPriorModelDirect()
    assert isinstance(model, PriorModel)


def test_lognormal_prior_model_direct_forward():
    """Test forward method of LogNormalPriorModelDirect."""
    model = LogNormalPriorModelDirect()
    
    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.rand(batch_size, n_genes)
    s_obs = torch.rand(batch_size, n_genes)
    
    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }
    
    # Initialize Pyro
    pyro.clear_param_store()
    
    # Call forward method
    with pyro.poutine.trace() as tr:
        result = model.forward(context)
    
    # Check that the context is updated with sampled parameters
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result
    assert "u_scale" in result
    assert "s_scale" in result
    assert "dt_switching" in result
    assert "t0" in result
    
    # Check that the parameters have the correct shape
    assert result["alpha"].shape == (n_genes,)
    assert result["beta"].shape == (n_genes,)
    assert result["gamma"].shape == (n_genes,)
    assert result["u_scale"].shape == (n_genes,)
    assert result["s_scale"].shape == (n_genes,)
    assert result["dt_switching"].shape == (n_genes,)
    assert result["t0"].shape == (n_genes,)
    
    # Check that the trace contains the expected sample sites
    trace = tr.trace
    assert "alpha" in trace.nodes
    assert "beta" in trace.nodes
    assert "gamma" in trace.nodes
    assert "u_scale" in trace.nodes
    assert "s_scale" in trace.nodes
    assert "dt_switching" in trace.nodes
    assert "t0" in trace.nodes


def test_lognormal_prior_model_direct_sample_parameters():
    """Test sample_parameters method of LogNormalPriorModelDirect."""
    model = LogNormalPriorModelDirect()
    
    # Call sample_parameters method
    params = model.sample_parameters(n_genes=4)
    
    # Check that the parameters are sampled
    assert "alpha" in params
    assert "beta" in params
    assert "gamma" in params
    assert "u_scale" in params
    assert "s_scale" in params
    assert "dt_switching" in params
    assert "t0" in params
    
    # Check that the parameters have the correct shape
    assert params["alpha"].shape == (4,)
    assert params["beta"].shape == (4,)
    assert params["gamma"].shape == (4,)
    assert params["u_scale"].shape == (4,)
    assert params["s_scale"].shape == (4,)
    assert params["dt_switching"].shape == (4,)
    assert params["t0"].shape == (4,)
