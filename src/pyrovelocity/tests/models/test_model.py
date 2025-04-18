"""
Integration tests for the PyroVelocityModel.
    
This module contains tests for the PyroVelocityModel class, verifying that
it correctly composes component models and implements the forward and guide
methods as expected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, Float

from pyrovelocity.models.components.base import (
    BaseDynamicsModel,
    BaseLikelihoodModel,
    BaseObservationModel,
    BasePriorModel,
    BaseInferenceGuide,
)
from pyrovelocity.models.components.dynamics import StandardDynamicsModel
from pyrovelocity.models.interfaces import (
    DynamicsModel as DynamicsModelProtocol,
    InferenceGuide as GuideModelProtocol,
    LikelihoodModel as LikelihoodModelProtocol,
    ObservationModel as ObservationModelProtocol,
    PriorModel as PriorModelProtocol,
)
from pyrovelocity.models.components.likelihoods import PoissonLikelihoodModel
from pyrovelocity.models.model import ModelState, PyroVelocityModel


# Mock implementations for testing
class MockDynamicsModel(BaseDynamicsModel):
    """Mock dynamics model for testing."""
    
    def __init__(self, name="mock_dynamics_model"):
        super().__init__(name=name)
        self.state = {}
    
    def forward(self, context):
        """Forward pass that just returns the input context."""
        # Extract required parameters from context or use defaults
        u = context.get("u", jnp.zeros((10, 5)))
        s = context.get("s", jnp.zeros((10, 5)))
        alpha = context.get("alpha", jnp.ones(5))
        beta = context.get("beta", jnp.ones(5))
        gamma = context.get("gamma", jnp.ones(5))
        
        # Add expected outputs to context
        context["u_expected"] = u
        context["s_expected"] = s
        
        return context


class MockPriorModel(BasePriorModel):
    """Mock prior model for testing."""
    
    def __init__(self, name="mock_prior_model"):
        super().__init__(name=name)
        self.state = {}
    
    def forward(self, context):
        """Forward pass that just returns the input context."""
        # Add rate parameters to context
        n_genes = context["x"].shape[1]
        context["alpha"] = jnp.ones(n_genes)
        context["beta"] = jnp.ones(n_genes)
        context["gamma"] = jnp.ones(n_genes)
        
        return context
    
    def _register_priors_impl(self, prefix=""):
        """Implementation of prior registration."""
        # No-op for testing
        pass
    
    def _sample_parameters_impl(self, prefix=""):
        """Implementation of parameter sampling."""
        # Return empty dict for testing
        return {}


class MockObservationModel(BaseObservationModel):
    """Mock observation model for testing."""
    
    def __init__(self, name="mock_observation_model"):
        super().__init__(name=name)
        self.state = {}
    
    def forward(self, context):
        """Forward pass that just returns the input context."""
        # Extract x from context
        x = context["x"]
        
        # Add u and s to context (for simplicity, just use x for both)
        context["u"] = x
        context["s"] = x
        
        return context
    
    def _prepare_data_impl(self, adata, **kwargs):
        """Implementation of data preparation."""
        # Return empty dict for testing
        return {}
    
    def _create_dataloaders_impl(self, data, **kwargs):
        """Implementation of dataloader creation."""
        # Return empty dict for testing
        return {}
    
    def _preprocess_batch_impl(self, batch):
        """Implementation of batch preprocessing."""
        # Return the input batch for testing
        return batch


class MockLikelihoodModel(BaseLikelihoodModel):
    """Mock likelihood model for testing."""
    
    def __init__(self, name="mock_likelihood_model"):
        super().__init__(name=name)
        self.state = {}
    
    def forward(self, context):
        """Forward pass that just returns the input context."""
        return context
    
    def _log_prob_impl(self, observations, predictions, scale_factors=None):
        """Implementation of log probability calculation."""
        # Return zeros for testing
        return jnp.zeros(observations.shape[0])
    
    def _sample_impl(self, predictions, scale_factors=None):
        """Implementation of sampling."""
        # Return the predictions for testing
        return predictions


class MockGuideModel(BaseInferenceGuide):
    """Mock guide model for testing."""
    
    def __init__(self, name="mock_guide_model"):
        super().__init__(name=name)
        self.state = {}
    
    def forward(self, context):
        """Forward pass that just returns the input context."""
        return context
    
    def __call__(self, model, *args, **kwargs):
        """Create a guide function for the given model."""
        def guide(*args, **kwargs):
            """Mock guide function."""
            return {}
        return guide
    
    def _setup_guide_impl(self, model, **kwargs):
        """Implementation of guide setup."""
        # No-op for testing
        pass
    
    def _sample_posterior_impl(self, model, guide, **kwargs):
        """Implementation of posterior sampling."""
        # Return empty dict for testing
        return {}


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create random data for testing
    key = jax.random.PRNGKey(42)
    n_cells = 10
    n_genes = 5
    n_times = 3
        
    # Generate random data
    x = jax.random.normal(key, (n_cells, n_genes))
    time_points = jnp.linspace(0, 1, n_times)
        
    return {
        "x": x,
        "time_points": time_points,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_times": n_times,
    }


@pytest.fixture
def component_models(sample_data):
    """Create component models for testing."""
    n_genes = sample_data["n_genes"]
        
    # Create component models with mock implementations
    dynamics_model = MockDynamicsModel()
    prior_model = MockPriorModel()
    likelihood_model = MockLikelihoodModel()
    observation_model = MockObservationModel()
    guide_model = MockGuideModel()
        
    return {
        "dynamics_model": dynamics_model,
        "prior_model": prior_model,
        "likelihood_model": likelihood_model,
        "observation_model": observation_model,
        "guide_model": guide_model,
    }


@pytest.fixture
def pyro_velocity_model(component_models):
    """Create a PyroVelocityModel instance for testing."""
    return PyroVelocityModel(
        dynamics_model=component_models["dynamics_model"],
        prior_model=component_models["prior_model"],
        likelihood_model=component_models["likelihood_model"],
        observation_model=component_models["observation_model"],
        guide_model=component_models["guide_model"],
    )


def test_model_initialization(pyro_velocity_model, component_models):
    """Test that the model initializes correctly with component models."""
    # Check that the model has the correct components
    assert pyro_velocity_model.dynamics_model == component_models["dynamics_model"]
    assert pyro_velocity_model.prior_model == component_models["prior_model"]
    assert pyro_velocity_model.likelihood_model == component_models["likelihood_model"]
    assert pyro_velocity_model.observation_model == component_models["observation_model"]
    assert pyro_velocity_model.guide_model == component_models["guide_model"]
        
    # Check that the model state is initialized correctly
    state = pyro_velocity_model.get_state()
    assert isinstance(state, ModelState)
    assert isinstance(state.dynamics_state, dict)
    assert isinstance(state.prior_state, dict)
    assert isinstance(state.likelihood_state, dict)
    assert isinstance(state.observation_state, dict)
    assert isinstance(state.guide_state, dict)


def test_model_forward(pyro_velocity_model, sample_data):
    """Test the forward method of the model."""
    # Run the forward method
    result = pyro_velocity_model.forward(
        x=sample_data["x"],
        time_points=sample_data["time_points"],
    )
        
    # Check that the result is a dictionary
    assert isinstance(result, dict)
        
    # Check that the result contains expected keys
    # The exact keys will depend on what each component adds to the context
    # but we can check for some common ones
    assert "x" in result
    assert "time_points" in result
    assert "u" in result
    assert "s" in result
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result


def test_model_guide(pyro_velocity_model, sample_data):
    """Test the guide method of the model."""
    # Run the guide method
    result = pyro_velocity_model.guide(
        x=sample_data["x"],
        time_points=sample_data["time_points"],
    )
        
    # Check that the result is a dictionary
    assert isinstance(result, dict)
        
    # Check that the result contains expected keys
    # The exact keys will depend on what the guide model adds to the context
    assert "x" in result
    assert "time_points" in result


def test_model_with_state(pyro_velocity_model):
    """Test the with_state method for immutable state updates."""
    # Get the current state
    original_state = pyro_velocity_model.get_state()
        
    # Create a new state with updated metadata
    new_metadata = {"test_key": "test_value"}
    new_state = ModelState(
        dynamics_state=original_state.dynamics_state,
        prior_state=original_state.prior_state,
        likelihood_state=original_state.likelihood_state,
        observation_state=original_state.observation_state,
        guide_state=original_state.guide_state,
        metadata=new_metadata,
    )
        
    # Create a new model with the updated state
    new_model = pyro_velocity_model.with_state(new_state)
        
    # Check that the new model has the updated state
    assert new_model.get_state().metadata == new_metadata
        
    # Check that the original model's state is unchanged
    assert pyro_velocity_model.get_state().metadata != new_metadata


def test_model_composition(component_models, sample_data):
    """Test that the model correctly composes component models."""
    # Create the model
    model = PyroVelocityModel(
        dynamics_model=component_models["dynamics_model"],
        prior_model=component_models["prior_model"],
        likelihood_model=component_models["likelihood_model"],
        observation_model=component_models["observation_model"],
        guide_model=component_models["guide_model"],
    )
        
    # Run the forward method
    result = model.forward(
        x=sample_data["x"],
        time_points=sample_data["time_points"],
    )
        
    # Check that the result contains contributions from each component
    assert "x" in result  # From input
    assert "time_points" in result  # From input
    assert "u" in result  # From observation model
    assert "s" in result  # From observation model
    assert "alpha" in result  # From prior model
    assert "beta" in result  # From prior model
    assert "gamma" in result  # From prior model
    assert "u_expected" in result  # From dynamics model
    assert "s_expected" in result  # From dynamics model