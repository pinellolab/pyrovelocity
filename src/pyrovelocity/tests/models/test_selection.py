"""
Tests for the model selection utilities.

This module contains tests for the ModelSelection, ModelEnsemble, and CrossValidator
classes and related utility functions for model selection and ensemble creation.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pyro
from anndata import AnnData

from pyrovelocity.models.comparison import (
    BayesianModelComparison,
    ComparisonResult,
)
from pyrovelocity.models.selection import (
    CrossValidator,
    ModelEnsemble,
    ModelSelection,
    SelectionCriterion,
    SelectionResult,
)
from pyrovelocity.models.model import ModelState, PyroVelocityModel

# Import mock models from test_comparison.py
from pyrovelocity.tests.models.test_comparison import (
    MockDynamicsModel,
    MockLikelihoodModel,
    MockPriorModel,
    MockObservationModel,
    MockGuideModel,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create random data for testing
    torch.manual_seed(42)
    n_cells = 10
    n_genes = 5
    
    # Generate random data
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
def mock_model(sample_data):
    """Create a mock PyroVelocityModel for testing."""
    dynamics_model = MockDynamicsModel()
    prior_model = MockPriorModel()
    likelihood_model = MockLikelihoodModel()
    observation_model = MockObservationModel()
    guide_model = MockGuideModel()
    
    return PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        observation_model=observation_model,
        guide_model=guide_model,
    )


@pytest.fixture
def mock_posterior_samples():
    """Create mock posterior samples for testing."""
    torch.manual_seed(42)
    
    # Create random posterior samples
    return {
        "alpha": torch.abs(torch.randn(100, 5)),
        "beta": torch.abs(torch.randn(100, 5)),
        "gamma": torch.abs(torch.randn(100, 5)),
    }


@pytest.fixture
def mock_models(mock_model):
    """Create a dictionary of mock models for testing."""
    # Create multiple instances of the same mock model for simplicity
    return {
        "model1": mock_model,
        "model2": mock_model,
        "model3": mock_model,
    }


@pytest.fixture
def mock_posterior_samples_dict(mock_posterior_samples):
    """Create a dictionary of mock posterior samples for testing."""
    # Create multiple copies of the same posterior samples for simplicity
    return {
        "model1": mock_posterior_samples,
        "model2": mock_posterior_samples,
        "model3": mock_posterior_samples,
    }


@pytest.fixture
def mock_comparison_result():
    """Create a mock ComparisonResult for testing."""
    values = {"model1": 100.0, "model2": 110.0, "model3": 90.0}
    standard_errors = {"model1": 5.0, "model2": 6.0, "model3": 4.0}
    
    # Create pairwise differences
    differences = {}
    model_names = list(values.keys())
    for model1 in model_names:
        differences[model1] = {}
        for model2 in model_names:
            if model1 != model2:
                diff = values[model1] - values[model2]
                differences[model1][model2] = diff
    
    return ComparisonResult(
        metric_name="WAIC",
        values=values,
        differences=differences,
        standard_errors=standard_errors,
    )


@pytest.fixture
def mock_selection_result(mock_comparison_result):
    """Create a mock SelectionResult for testing."""
    return SelectionResult(
        selected_model_name="model3",  # model3 has lowest WAIC
        criterion=SelectionCriterion.WAIC,
        comparison_result=mock_comparison_result,
        is_significant=True,
        significance_threshold=2.0,
    )


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    # Create a simple AnnData object
    n_cells = 10
    n_genes = 5
    X = np.random.randn(n_cells, n_genes)
    
    # Create cell metadata
    obs = pd.DataFrame({
        "cell_type": np.random.choice(["A", "B", "C"], size=n_cells),
        "time_point": np.random.choice([0, 1, 2], size=n_cells),
    })
    
    return AnnData(X=X, obs=obs)


def test_selection_criterion_enum():
    """Test SelectionCriterion enum values."""
    assert SelectionCriterion.WAIC.value == "waic"
    assert SelectionCriterion.LOO.value == "loo"
    assert SelectionCriterion.BAYES_FACTOR.value == "bayes_factor"
    assert SelectionCriterion.CV_LIKELIHOOD.value == "cv_likelihood"
    assert SelectionCriterion.CV_ERROR.value == "cv_error"


def test_selection_result_initialization(mock_comparison_result):
    """Test initialization of SelectionResult."""
    result = SelectionResult(
        selected_model_name="model1",
        criterion=SelectionCriterion.WAIC,
        comparison_result=mock_comparison_result,
        is_significant=True,
        significance_threshold=2.0,
    )
    
    # Check attributes
    assert result.selected_model_name == "model1"
    assert result.criterion == SelectionCriterion.WAIC
    assert result.comparison_result == mock_comparison_result
    assert result.is_significant is True
    assert result.significance_threshold == 2.0
    assert result.metadata is None


def test_selection_result_to_dataframe(mock_selection_result):
    """Test to_dataframe method of SelectionResult."""
    df = mock_selection_result.to_dataframe()
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert "model" in df.columns
    assert "WAIC" in df.columns
    assert "WAIC_se" in df.columns
    assert "selected" in df.columns
    assert "significance" in df.columns
    
    # Check values
    assert len(df) == 3  # Three models
    assert df.loc[df["model"] == "model3", "selected"].iloc[0] == True
    assert df.loc[df["model"] != "model3", "selected"].all() == False
    assert df["significance"].all() == True  # All rows have significance=True


@patch.object(BayesianModelComparison, "compare_models")
def test_model_selection_waic(mock_compare_models, mock_models, mock_posterior_samples_dict, sample_data, mock_comparison_result):
    """Test select_model method with WAIC criterion."""
    # Mock compare_models to return a predefined ComparisonResult
    mock_compare_models.return_value = mock_comparison_result
    
    # Create ModelSelection instance
    selection = ModelSelection()
    
    # Select model using WAIC
    result = selection.select_model(
        models=mock_models,
        posterior_samples=mock_posterior_samples_dict,
        data=sample_data,
        criterion=SelectionCriterion.WAIC,
    )
    
    # Check that compare_models was called with the right arguments
    mock_compare_models.assert_called_once_with(
        models=mock_models,
        posterior_samples=mock_posterior_samples_dict,
        data=sample_data,
        metric="waic",
    )
    
    # Check result
    assert isinstance(result, SelectionResult)
    assert result.selected_model_name == "model3"  # model3 has lowest WAIC
    assert result.criterion == SelectionCriterion.WAIC
    assert result.comparison_result == mock_comparison_result


@patch.object(BayesianModelComparison, "compare_models")
def test_model_selection_loo(mock_compare_models, mock_models, mock_posterior_samples_dict, sample_data, mock_comparison_result):
    """Test select_model method with LOO criterion."""
    # Mock compare_models to return a predefined ComparisonResult
    mock_compare_models.return_value = mock_comparison_result
    
    # Create ModelSelection instance
    selection = ModelSelection()
    
    # Select model using LOO
    result = selection.select_model(
        models=mock_models,
        posterior_samples=mock_posterior_samples_dict,
        data=sample_data,
        criterion=SelectionCriterion.LOO,
    )
    
    # Check that compare_models was called with the right arguments
    mock_compare_models.assert_called_once_with(
        models=mock_models,
        posterior_samples=mock_posterior_samples_dict,
        data=sample_data,
        metric="loo",
    )
    
    # Check result
    assert isinstance(result, SelectionResult)
    assert result.selected_model_name == "model3"  # model3 has lowest LOO
    assert result.criterion == SelectionCriterion.LOO
    assert result.comparison_result == mock_comparison_result


@patch.object(BayesianModelComparison, "compare_models_bayes_factors")
def test_model_selection_bayes_factor(mock_compare_models_bf, mock_models, mock_posterior_samples_dict, sample_data):
    """Test select_model method with Bayes factor criterion."""
    # Create a Bayes factor comparison result
    bf_values = {"model1": 2.0, "model2": 0.5, "model3": 1.0}
    bf_result = ComparisonResult(
        metric_name="Bayes Factor",
        values=bf_values,
    )
    
    # Mock compare_models_bayes_factors to return a predefined ComparisonResult
    mock_compare_models_bf.return_value = bf_result
    
    # Create ModelSelection instance
    selection = ModelSelection()
    
    # Select model using Bayes factor
    result = selection.select_model(
        models=mock_models,
        posterior_samples=mock_posterior_samples_dict,
        data=sample_data,
        criterion=SelectionCriterion.BAYES_FACTOR,
    )
    
    # Check that compare_models_bayes_factors was called with the right arguments
    mock_compare_models_bf.assert_called_once_with(
        models=mock_models,
        data=sample_data,
    )
    
    # Check result
    assert isinstance(result, SelectionResult)
    assert result.selected_model_name == "model1"  # model1 has highest Bayes factor
    assert result.criterion == SelectionCriterion.BAYES_FACTOR
    assert result.comparison_result == bf_result


def test_model_ensemble_initialization(mock_models):
    """Test initialization of ModelEnsemble."""
    # Test with default equal weights
    ensemble = ModelEnsemble(models=mock_models)
    
    # Check attributes
    assert ensemble.models == mock_models
    assert isinstance(ensemble.weights, dict)
    assert set(ensemble.weights.keys()) == set(mock_models.keys())
    assert all(np.isclose(w, 1.0 / len(mock_models)) for w in ensemble.weights.values())
    
    # Test with custom weights
    weights = {"model1": 0.5, "model2": 0.3, "model3": 0.2}
    ensemble = ModelEnsemble(models=mock_models, weights=weights)
    
    # Check attributes
    assert ensemble.models == mock_models
    assert ensemble.weights == weights
    
    # Test with invalid weights (missing model)
    invalid_weights = {"model1": 0.5, "model2": 0.5}
    with pytest.raises(ValueError):
        ModelEnsemble(models=mock_models, weights=invalid_weights)
    
    # Test with invalid weights (extra model)
    invalid_weights = {"model1": 0.4, "model2": 0.3, "model3": 0.2, "model4": 0.1}
    with pytest.raises(ValueError):
        ModelEnsemble(models=mock_models, weights=invalid_weights)


@patch.object(MockDynamicsModel, "forward")
@patch.object(MockObservationModel, "forward")
def test_model_ensemble_predict(mock_obs_forward, mock_dyn_forward, mock_models, mock_posterior_samples_dict, sample_data):
    """Test predict method of ModelEnsemble."""
    # Mock observation model forward to return the input context
    mock_obs_forward.side_effect = lambda context: context
    
    # Mock dynamics model forward to return a context with predictions
    def mock_dyn_forward_impl(context):
        # Extract parameters from context
        parameters = context.get("parameters", {})
        
        # Add predictions to context
        context["predictions"] = torch.ones((10, 5)) * parameters.get("alpha", 1.0)[0]
        
        return context
    
    mock_dyn_forward.side_effect = mock_dyn_forward_impl
    
    # Create ModelEnsemble instance
    ensemble = ModelEnsemble(models=mock_models)
    
    # Convert torch tensors to jax arrays for compatibility with jaxtyping
    import jax.numpy as jnp
    x_jax = jnp.array(sample_data["x"].numpy())
    time_points_jax = jnp.array(sample_data["time_points"].numpy())
    
    # Generate predictions
    predictions = ensemble.predict(
        x=x_jax,
        time_points=time_points_jax,
        posterior_samples=mock_posterior_samples_dict,
    )
    
    # Check result
    assert "ensemble_mean" in predictions
    assert "ensemble_std" in predictions
    assert "model_predictions" in predictions
    assert "model_weights" in predictions
    
    # Check shapes
    assert predictions["ensemble_mean"].shape == (10, 5)
    assert predictions["ensemble_std"].shape == (10, 5)
    assert len(predictions["model_predictions"]) == 3  # Three models
    assert len(predictions["model_weights"]) == 3  # Three models


def test_model_ensemble_from_selection_result(mock_models, mock_selection_result):
    """Test from_selection_result class method of ModelEnsemble."""
    # Create ensemble with equal weights
    ensemble = ModelEnsemble.from_selection_result(
        selection_result=mock_selection_result,
        models=mock_models,
        use_weights=False,
    )
    
    # Check attributes
    assert ensemble.models == mock_models
    assert isinstance(ensemble.weights, dict)
    assert set(ensemble.weights.keys()) == set(mock_models.keys())
    assert all(np.isclose(w, 1.0 / len(mock_models)) for w in ensemble.weights.values())
    
    # Create ensemble with metric-based weights
    ensemble = ModelEnsemble.from_selection_result(
        selection_result=mock_selection_result,
        models=mock_models,
        use_weights=True,
    )
    
    # Check attributes
    assert ensemble.models == mock_models
    assert isinstance(ensemble.weights, dict)
    assert set(ensemble.weights.keys()) == set(mock_models.keys())
    
    # For WAIC, lower is better, so model3 (90.0) should have highest weight
    assert ensemble.weights["model3"] > ensemble.weights["model1"]
    assert ensemble.weights["model1"] > ensemble.weights["model2"]


def test_model_ensemble_from_top_k_models(mock_models, mock_selection_result):
    """Test from_top_k_models class method of ModelEnsemble."""
    # Create ensemble with top 2 models
    ensemble = ModelEnsemble.from_top_k_models(
        selection_result=mock_selection_result,
        models=mock_models,
        k=2,
        use_weights=False,
    )
    
    # Check attributes
    assert len(ensemble.models) == 2
    assert "model3" in ensemble.models  # Best model
    assert "model1" in ensemble.models  # Second best model
    assert "model2" not in ensemble.models  # Worst model
    assert all(np.isclose(w, 0.5) for w in ensemble.weights.values())  # Equal weights
    
    # Create ensemble with top 2 models and metric-based weights
    ensemble = ModelEnsemble.from_top_k_models(
        selection_result=mock_selection_result,
        models=mock_models,
        k=2,
        use_weights=True,
    )
    
    # Check attributes
    assert len(ensemble.models) == 2
    assert "model3" in ensemble.models  # Best model
    assert "model1" in ensemble.models  # Second best model
    assert "model2" not in ensemble.models  # Worst model
    assert ensemble.weights["model3"] > ensemble.weights["model1"]  # model3 has better WAIC


def test_cross_validator_initialization():
    """Test initialization of CrossValidator."""
    # Test with default parameters
    cv = CrossValidator()
    assert cv.n_splits == 5
    assert cv.stratify_by is None
    assert cv.random_state == 42
    
    # Test with custom parameters
    cv = CrossValidator(n_splits=10, stratify_by="cell_type", random_state=123)
    assert cv.n_splits == 10
    assert cv.stratify_by == "cell_type"
    assert cv.random_state == 123


def test_cross_validator_get_cv_splitter(mock_adata):
    """Test _get_cv_splitter method of CrossValidator."""
    from sklearn.model_selection import KFold, StratifiedKFold
    
    # Test with no stratification
    cv = CrossValidator()
    splitter = cv._get_cv_splitter(mock_adata)
    assert isinstance(splitter, KFold)
    assert splitter.n_splits == 5
    
    # Test with stratification
    cv = CrossValidator(stratify_by="cell_type")
    splitter = cv._get_cv_splitter(mock_adata)
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 5
    
    # Test with invalid stratification column
    cv = CrossValidator(stratify_by="invalid_column")
    with pytest.raises(ValueError):
        cv._get_cv_splitter(mock_adata)


@patch("pyro.infer.SVI")
@patch.object(MockGuideModel, "_sample_posterior_impl")
@patch.object(MockLikelihoodModel, "log_prob")
def test_cross_validator_cross_validate_likelihood(mock_log_prob, mock_sample_posterior, mock_svi, mock_model, sample_data, mock_adata):
    """Test cross_validate_likelihood method of CrossValidator."""
    # Mock SVI step to return a loss value
    mock_svi_instance = MagicMock()
    mock_svi_instance.step.return_value = 100.0
    mock_svi.return_value = mock_svi_instance
    
    # Mock _sample_posterior_impl to return posterior samples
    mock_sample_posterior.return_value = {
        "alpha": torch.ones((100, 5)),
        "beta": torch.ones((100, 5)),
        "gamma": torch.ones((100, 5)),
    }
    
    # Mock log_prob to return log likelihood values
    mock_log_prob.return_value = jnp.ones((10,)) * 2.0
    
    # Create CrossValidator instance
    cv = CrossValidator(n_splits=2)  # Use 2 splits for faster testing
    
    # Run cross-validation
    scores = cv.cross_validate_likelihood(
        model=mock_model,
        data=sample_data,
        adata=mock_adata,
    )
    
    # Check result
    assert isinstance(scores, list)
    assert len(scores) == 2  # Two folds
    assert all(np.isclose(score, 2.0) for score in scores)  # All scores should be 2.0


@patch("pyro.infer.SVI")
@patch.object(MockGuideModel, "_sample_posterior_impl")
def test_cross_validator_cross_validate_error(mock_sample_posterior, mock_svi, mock_model, sample_data, mock_adata):
    """Test cross_validate_error method of CrossValidator."""
    # Mock SVI step to return a loss value
    mock_svi_instance = MagicMock()
    mock_svi_instance.step.return_value = 100.0
    mock_svi.return_value = mock_svi_instance
    
    # Mock _sample_posterior_impl to return posterior samples
    mock_sample_posterior.return_value = {
        "alpha": torch.ones((100, 5)),
        "beta": torch.ones((100, 5)),
        "gamma": torch.ones((100, 5)),
    }
    
    # Create CrossValidator instance
    cv = CrossValidator(n_splits=2)  # Use 2 splits for faster testing
    
    # Run cross-validation
    scores = cv.cross_validate_error(
        model=mock_model,
        data=sample_data,
        adata=mock_adata,
    )
    
    # Check result
    assert isinstance(scores, list)
    assert len(scores) == 2  # Two folds
    assert all(score > 0 for score in scores)  # All scores should be positive