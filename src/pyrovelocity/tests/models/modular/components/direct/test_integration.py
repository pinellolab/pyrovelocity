"""Integration tests for Protocol-First component implementations."""

import pytest
import pyro
import torch

from pyrovelocity.models.modular.components.direct.dynamics import StandardDynamicsModelDirect
from pyrovelocity.models.modular.components.direct.guides import AutoGuideFactoryDirect
from pyrovelocity.models.modular.components.direct.likelihoods import PoissonLikelihoodModelDirect
from pyrovelocity.models.modular.components.direct.observations import StandardObservationModelDirect
from pyrovelocity.models.modular.components.direct.priors import LogNormalPriorModelDirect
from pyrovelocity.models.modular.model import PyroVelocityModel


@pytest.fixture
def simple_data():
    """Create simple data for testing."""
    # Create random data
    batch_size = 3
    n_genes = 4
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    
    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }


def test_protocol_first_components_integration(simple_data):
    """Test that all Protocol-First components work together."""
    # Create components
    dynamics_model = StandardDynamicsModelDirect()
    prior_model = LogNormalPriorModelDirect()
    likelihood_model = PoissonLikelihoodModelDirect()
    observation_model = StandardObservationModelDirect()
    guide_model = AutoGuideFactoryDirect(guide_type="AutoNormal")
    
    # Create context
    context = {
        "u_obs": simple_data["u_obs"],
        "s_obs": simple_data["s_obs"],
    }
    
    # Initialize Pyro
    pyro.clear_param_store()
    
    # Call each component's forward method
    context = observation_model.forward(context)
    context = prior_model.forward(context)
    context = dynamics_model.forward(context)
    context = likelihood_model.forward(context)
    
    # Check that the context contains all expected keys
    assert "u_obs" in context
    assert "s_obs" in context
    assert "u_scale" in context
    assert "s_scale" in context
    assert "alpha" in context
    assert "beta" in context
    assert "gamma" in context
    assert "u_expected" in context
    assert "s_expected" in context
    assert "u_dist" in context
    assert "s_dist" in context


def test_protocol_first_components_with_model(simple_data):
    """Test that Protocol-First components work with PyroVelocityModel."""
    # Create components
    dynamics_model = StandardDynamicsModelDirect()
    prior_model = LogNormalPriorModelDirect()
    likelihood_model = PoissonLikelihoodModelDirect()
    observation_model = StandardObservationModelDirect()
    guide_model = AutoGuideFactoryDirect(guide_type="AutoNormal")
    
    # Create the full model
    model = PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        observation_model=observation_model,
        guide_model=guide_model,
    )
    
    # Initialize Pyro
    pyro.clear_param_store()
    
    # Call the model's forward method
    with pyro.poutine.trace() as tr:
        result = model.forward(
            u_obs=simple_data["u_obs"],
            s_obs=simple_data["s_obs"],
        )
    
    # Check that the result contains expected keys
    assert "u_dist" in result
    assert "s_dist" in result
    
    # Check that the trace contains expected sample sites
    trace = tr.trace
    assert "alpha" in trace.nodes
    assert "beta" in trace.nodes
    assert "gamma" in trace.nodes
    assert "u_obs" in trace.nodes
    assert "s_obs" in trace.nodes
