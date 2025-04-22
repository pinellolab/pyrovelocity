"""
Shared fixtures for model selection tests.

This module provides fixtures that are shared between test files for the selection module,
facilitating the testing of model selection functionality in the PyroVelocity framework.
"""

from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.modular.comparison import (
    BayesianModelComparison,
    ComparisonResult,
)
from pyrovelocity.models.modular.components.base import (
    BaseDynamicsModel,
    BaseInferenceGuide,
    BaseLikelihoodModel,
    BaseObservationModel,
    BasePriorModel,
)
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel
from pyrovelocity.models.modular.selection import (
    CrossValidator,
    ModelEnsemble,
    ModelSelection,
    SelectionCriterion,
    SelectionResult,
)


# Mock implementations for testing
class MockDynamicsModel(BaseDynamicsModel):
    """Mock dynamics model for testing."""

    def __init__(self, name="mock_dynamics_model"):
        super().__init__(name=name)
        self.state = {}

    def forward(self, context):
        """Forward pass that applies the dynamics model to input data."""
        # Extract parameters from context
        parameters = context.get("parameters", {})
        alpha = parameters.get("alpha", torch.tensor(1.0))
        beta = parameters.get("beta", torch.tensor(1.0))
        gamma = parameters.get("gamma", torch.tensor(1.0))
        
        # Extract state from context
        u = context.get("u", torch.ones((10, 5)))
        s = context.get("s", torch.ones((10, 5)))
        
        # Apply forward transform
        u_new, s_new = self._forward_impl(
            u=u,
            s=s,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        
        # Create predictions dictionary
        predictions = {
            "u": u_new,
            "s": s_new,
            "mu": u_new,  # Example: use unspliced as prediction for mu
            "ms": s_new,  # Example: use spliced as prediction for ms
        }
        
        # Add predictions to context
        context["predictions"] = predictions
        
        return context

    def _forward_impl(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: torch.Tensor = None,
        t: torch.Tensor = None,
    ):
        """Implementation of the forward pass."""
        # Simple dummy dynamics for testing
        u_new = u + 0.1
        s_new = s + 0.2
        return u_new, s_new

    def _steady_state_impl(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: torch.Tensor = None,
    ):
        """Implementation of steady state calculation."""
        # Simple dummy steady state for testing
        u_ss = alpha / beta
        s_ss = alpha / gamma
        return u_ss, s_ss

    def _predict_future_states_impl(
        self,
        current_state,
        time_delta: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: torch.Tensor = None,
    ):
        """Implementation of future state prediction."""
        # Simple dummy future state prediction for testing
        u_current, s_current = current_state
        u_future = u_current + 0.1 * time_delta
        s_future = s_current + 0.2 * time_delta
        return u_future, s_future


class MockLikelihoodModel(BaseLikelihoodModel):
    """Mock likelihood model for testing."""

    def __init__(self, name="mock_likelihood_model"):
        super().__init__(name=name)
        self.state = {}

    def forward(self, context):
        """Forward pass that just returns the input context."""
        return context

    def _log_prob_impl(
        self, 
        observations: Float[Array, "batch_size genes"],
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None
    ) -> Float[Array, "batch_size"]:
        """Implementation of log probability calculation."""
        # Simple mock implementation that returns ones
        batch_size = observations.shape[0]
        return torch.ones(batch_size) * (-1.0)

    def _sample_impl(
        self,
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None
    ) -> Float[Array, "batch_size genes"]:
        """Implementation of sampling."""
        # Add small Gaussian noise
        noise = torch.randn(*predictions.shape) * 0.1
        return predictions + noise


class MockPriorModel(BasePriorModel):
    """Mock prior model for testing."""

    def __init__(self, name="mock_prior_model"):
        super().__init__(name=name)
        self.state = {}

    def forward(self, context):
        """Forward pass that adds parameters to the context."""
        # Add parameters to context
        context["parameters"] = {
            "alpha": torch.tensor(1.0),
            "beta": torch.tensor(1.0),
            "gamma": torch.tensor(1.0),
        }

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
        """Forward pass that adds observations to the context."""
        # Add observations to context
        context["observations"] = context.get("x", torch.ones((10, 5)))

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
        # No-op for testing
        return batch


class MockGuideModel(BaseInferenceGuide):
    """Mock guide model for testing."""

    def __init__(self, name="mock_guide_model"):
        super().__init__(name=name)
        self.state = {}

    def forward(self, context):
        """Forward pass that just returns the input context."""
        return context

    def __call__(self, model, *args, **kwargs):
        """Return a guide function that can be used by Pyro."""
        def guide(*args, **kwargs):
            """Mock guide function."""
            # No-op for testing
            return None
        
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
    n_cells = 10
    n_genes = 5
    
    x = torch.randn((n_cells, n_genes))
    time_points = torch.linspace(0, 1, 3)
    observations = torch.abs(torch.randn((n_cells, n_genes)))

    return {
        "x": x,
        "time_points": time_points,
        "observations": observations,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def mock_adata():
    """Create a real AnnData object for testing."""
    import anndata
    import pandas as pd
    
    # Generate random data
    n_obs = 10
    n_vars = 5
    X = np.random.randn(n_obs, n_vars)
    
    # Create observation annotations
    obs = pd.DataFrame({
        "cell_type": np.random.choice(["type1", "type2", "type3"], size=n_obs),
        "time": np.random.rand(n_obs),
    }, index=[f"cell_{i}" for i in range(n_obs)])
    
    # Create variable annotations
    var = pd.DataFrame({
        "gene_type": np.random.choice(["coding", "non-coding"], size=n_vars),
    }, index=[f"gene_{i}" for i in range(n_vars)])
    
    # Create AnnData object
    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=var,
        uns={"dataset_info": "test_dataset"},
    )
    
    return adata


@pytest.fixture
def mock_model(sample_data):
    """Create a mock PyroVelocityModel for testing."""
    dynamics_model = MockDynamicsModel()
    prior_model = MockPriorModel()
    likelihood_model = MockLikelihoodModel()
    observation_model = MockObservationModel()
    guide_model = MockGuideModel()

    model = MockPyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        observation_model=observation_model,
        guide_model=guide_model,
        name="mock_model"
    )
    
    # Add test result attributes
    model.result = MagicMock()
    model.result.posterior_samples = {
        "alpha": torch.ones(10, 3),
        "beta": torch.ones(10, 3),
        "gamma": torch.ones(10, 3),
    }
    model.result.log_likelihood = torch.ones(10, 20)
    
    return model


@pytest.fixture
def multiple_models(sample_data):
    """Create multiple mock PyroVelocityModels for testing."""
    models = []
    for i in range(3):
        dynamics_model = MockDynamicsModel(name=f"dynamics_{i}")
        prior_model = MockPriorModel(name=f"prior_{i}")
        likelihood_model = MockLikelihoodModel(name=f"likelihood_{i}")
        observation_model = MockObservationModel(name=f"observation_{i}")
        guide_model = MockGuideModel(name=f"guide_{i}")

        model = MockPyroVelocityModel(
            dynamics_model=dynamics_model,
            prior_model=prior_model,
            likelihood_model=likelihood_model,
            observation_model=observation_model,
            guide_model=guide_model,
            name=f"model_{i}"
        )
        
        # Add test result attributes
        model.result = MagicMock()
        model.result.posterior_samples = {
            "alpha": torch.ones(10, 3) * (i + 1),
            "beta": torch.ones(10, 3) * (i + 1),
            "gamma": torch.ones(10, 3) * (i + 1),
        }
        model.result.log_likelihood = torch.ones(10, 20) * (i + 1)
        
        models.append(model)
    
    return models


@pytest.fixture
def mock_comparison_result(multiple_models):
    """Create a mock ComparisonResult for testing."""
    names = [model.name for model in multiple_models]
    
    # Create values dict for WAIC
    waic_values = {names[i]: float(i) for i in range(len(names))}
    
    # Create values dict for LOO
    loo_values = {names[i]: float(i * 2) for i in range(len(names))}
    
    # Create Bayes factors
    bayes_factors = {
        f"{names[i]}:{names[j]}": float(i / (j + 1))
        for i in range(len(names))
        for j in range(len(names))
        if i != j
    }
    
    # Create WAIC comparison result
    waic_result = ComparisonResult(
        metric_name="WAIC",
        values=waic_values,
        differences=None,
        standard_errors=None,
    )
    
    # Create LOO comparison result
    loo_result = ComparisonResult(
        metric_name="LOO",
        values=loo_values,
        differences=None,
        standard_errors=None,
    )
    
    # Return WAIC result as default
    return waic_result


@pytest.fixture
def mock_selection_result(multiple_models, mock_comparison_result):
    """Create a mock SelectionResult for testing."""
    return SelectionResult(
        selected_model_name="model_0",
        criterion=SelectionCriterion.WAIC,
        comparison_result=mock_comparison_result,
        is_significant=True,
        significance_threshold=2.0,
        metadata={"weights": [0.5, 0.3, 0.2]}
    )


@pytest.fixture
def mock_cross_validation_result():
    """Create a mock cross-validation result for testing."""
    return {
        "model_0": {
            "fold_0": {"metric_1": 0.9, "metric_2": 0.8},
            "fold_1": {"metric_1": 0.85, "metric_2": 0.75},
        },
        "model_1": {
            "fold_0": {"metric_1": 0.8, "metric_2": 0.7},
            "fold_1": {"metric_1": 0.75, "metric_2": 0.65},
        },
    }


@pytest.fixture
def model_selection(multiple_models):
    """Create a ModelSelection instance for testing."""
    return ModelSelection(models=multiple_models)


@pytest.fixture
def model_ensemble(multiple_models):
    """Create a ModelEnsemble instance for testing."""
    models_dict = {model.name: model for model in multiple_models}
    weights_dict = {model.name: weight for model, weight in zip(multiple_models, [0.5, 0.3, 0.2])}
    return ModelEnsemble(models=models_dict, weights=weights_dict)


@pytest.fixture
def cross_validator(multiple_models):
    """Create a CrossValidator instance for testing."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    return CrossValidator(models=models_dict, n_splits=2, test_size=0.2)


# Mock PyroVelocityModel that allows setting the name property for testing
class MockPyroVelocityModel(PyroVelocityModel):
    """Mock PyroVelocityModel that allows setting the name property for testing."""
    
    def __init__(
        self,
        dynamics_model,
        prior_model,
        likelihood_model,
        observation_model,
        guide_model,
        state=None,
        name="mock_model"
    ):
        super().__init__(
            dynamics_model=dynamics_model,
            prior_model=prior_model,
            likelihood_model=likelihood_model,
            observation_model=observation_model,
            guide_model=guide_model,
            state=state,
        )
        self._name = name
    
    @property
    def name(self):
        """Return the custom model name."""
        return self._name 