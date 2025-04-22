"""
Tests for the model comparison functionality in PyroVelocity.

This module tests the BayesianModelComparison class, which is responsible for
comparing Bayesian models using various information criteria and metrics.
"""

from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Import the MockPyroVelocityModel from conftest instead of defining it here
from conftest import MockPyroVelocityModel

from pyrovelocity.models.modular.comparison import (
    BayesianModelComparison,
    ComparisonResult,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


def test_bayesian_model_comparison_init():
    """Test the initialization of BayesianModelComparison."""
    # Initialize with default parameters
    comparison = BayesianModelComparison()
    assert comparison.name == "model_comparison"  # Default name
    
    # Initialize with custom name
    comparison = BayesianModelComparison(name="custom_comparison")
    assert comparison.name == "custom_comparison"


def test_compute_waic(multiple_models):
    """Test the compute_waic method."""
    comparison = BayesianModelComparison()
    
    # Mock arviz.waic to return a predefined result
    mock_waic_result = MagicMock()
    mock_waic_result.waic = 10.5
    mock_waic_result.p_waic = 3.5
    mock_waic_result.waic_se = 1.2
    mock_waic_result.p_waic_se = 0.5
    mock_waic_result.pointwise = torch.ones(10)
    
    with patch("arviz.waic", return_value=mock_waic_result), \
         patch.object(BayesianModelComparison, "_extract_log_likelihood", return_value=torch.ones(10, 10)):
        # Compute WAIC for a model
        model = multiple_models[0]
        
        # Create mock posterior samples and data
        posterior_samples = {
            "alpha": torch.ones(10, 3),
            "beta": torch.ones(10, 3),
            "gamma": torch.ones(10, 3),
        }
        
        data = {
            "observations": torch.ones(10, 5),
            "time_points": torch.linspace(0, 1, 3),
        }
        
        # Call compute_waic with all required parameters
        waic_value, pointwise_waic = comparison.compute_waic(
            model=model, 
            posterior_samples=posterior_samples,
            data=data,
            pointwise=True
        )
        
        # Verify the correct values were returned
        assert waic_value == 10.5
        assert torch.allclose(torch.tensor(pointwise_waic), torch.ones(10))


def test_compute_loo(multiple_models):
    """Test the compute_loo method."""
    comparison = BayesianModelComparison()
    
    # Mock arviz.loo to return a predefined result
    mock_loo_result = MagicMock()
    mock_loo_result.loo = 15.2
    mock_loo_result.p_loo = 4.8
    mock_loo_result.loo_se = 2.1
    mock_loo_result.p_loo_se = 0.9
    mock_loo_result.pointwise = torch.ones(10)
    
    with patch("arviz.loo", return_value=mock_loo_result), \
         patch.object(BayesianModelComparison, "_extract_log_likelihood", return_value=torch.ones(10, 10)):
        # Compute LOO for a model
        model = multiple_models[0]
        
        # Create mock posterior samples and data
        posterior_samples = {
            "alpha": torch.ones(10, 3),
            "beta": torch.ones(10, 3),
            "gamma": torch.ones(10, 3),
        }
        
        data = {
            "observations": torch.ones(10, 5),
            "time_points": torch.linspace(0, 1, 3),
        }
        
        # Call compute_loo with all required parameters
        loo_value, pointwise_loo = comparison.compute_loo(
            model=model, 
            posterior_samples=posterior_samples,
            data=data,
            pointwise=True
        )
        
        # Verify the correct values were returned
        assert loo_value == 15.2
        assert torch.allclose(torch.tensor(pointwise_loo), torch.ones(10))


def test_extract_log_likelihood(multiple_models):
    """Test the _extract_log_likelihood method."""
    comparison = BayesianModelComparison()
    
    # Create mock components
    dynamics_model = MagicMock()
    dynamics_model.forward.return_value = {"predictions": torch.ones(10, 5)}
    
    likelihood_model = MagicMock()
    likelihood_model.log_prob.return_value = torch.ones(10)
    
    # Get a model from the fixture and replace its components with our mocks
    model = multiple_models[0]
    model.dynamics_model = dynamics_model
    model.likelihood_model = likelihood_model
    
    # Create mock posterior samples and data
    posterior_samples = {
        "alpha": torch.ones(10, 3),
        "beta": torch.ones(10, 3),
        "gamma": torch.ones(10, 3),
    }
    
    data = {
        "observations": torch.ones(10, 5),
        "time_points": torch.linspace(0, 1, 3),
    }
    
    # Extract log likelihood
    log_likelihood = comparison._extract_log_likelihood(
        model, 
        posterior_samples=posterior_samples,
        data=data
    )
    
    # Verify that the result is a tensor with the correct shape
    assert isinstance(log_likelihood, torch.Tensor)
    assert log_likelihood.shape[0] == 10  # Number of samples
    assert log_likelihood.shape[1] == 10  # Number of observations


def test_compute_bayes_factor(multiple_models):
    """Test the compute_bayes_factor method."""
    comparison = BayesianModelComparison()
    
    # Use models from the fixture
    model1 = multiple_models[0]
    model2 = multiple_models[1]
    
    # Create mock data
    data = {
        "observations": torch.ones(10, 5),
        "time_points": torch.linspace(0, 1, 3),
    }
    
    # Mock the _compute_log_marginal_likelihood method to return predictable values
    with patch.object(comparison, "_compute_log_marginal_likelihood") as mock_compute:
        # Set up mock return values for marginal likelihoods
        mock_compute.side_effect = [10.0, 5.0]  # log_ml1, log_ml2
        
        # Compute Bayes factor
        bf = comparison.compute_bayes_factor(
            model1=model1,
            model2=model2,
            data=data
        )
        
        # Expected Bayes factor: exp(log_ml1 - log_ml2) = exp(10.0 - 5.0) = exp(5.0)
        expected_bf = np.exp(5.0)
        assert bf == pytest.approx(expected_bf)


def test_compare_models(multiple_models):
    """Test the compare_models method."""
    comparison = BayesianModelComparison()
    
    # Use models from fixture
    models = {f"model_{i}": model for i, model in enumerate(multiple_models)}
    
    # Create posterior samples dictionary
    posterior_samples = {}
    for i, model_name in enumerate(models.keys()):
        posterior_samples[model_name] = {
            "alpha": torch.ones(10, 3) * (i + 1),
            "beta": torch.ones(10, 3) * (i + 1),
            "gamma": torch.ones(10, 3) * (i + 1),
        }
    
    # Create data dictionary
    data = {
        "observations": torch.ones(10, 5),
        "time_points": torch.linspace(0, 1, 3),
    }
    
    # Mock arviz.waic to return different values for each model
    mock_waic_results = []
    for i in range(3):
        mock_result = MagicMock()
        mock_result.waic = float(i)
        mock_result.waic_se = float(i) * 0.1
        mock_waic_results.append(mock_result)
    
    # Patch arviz functions with our mocks
    with patch("arviz.waic", side_effect=mock_waic_results), \
         patch("arviz.data.convert_to_dataset", return_value=MagicMock()), \
         patch.object(BayesianModelComparison, "_extract_log_likelihood", return_value=torch.ones(10, 10)):
        
        # Compare models
        result = comparison.compare_models(
            models=models,
            posterior_samples=posterior_samples,
            data=data,
            metric="waic"
        )
        
        # Verify the result is a ComparisonResult
        assert isinstance(result, ComparisonResult)
        
        # Verify the metric name
        assert result.metric_name == "WAIC"
        
        # Verify model values
        assert len(result.values) == len(models)
        for i, model_name in enumerate(models.keys()):
            assert result.values[model_name] == float(i)


def test_comparison_result_properties():
    """Test the properties of the ComparisonResult class."""
    # Create a mock comparison result
    result = ComparisonResult(
        metric_name="WAIC",
        values={"model_0": 0, "model_1": 1, "model_2": 2},
        differences={"model_0": {"model_1": -1, "model_2": -2},
                     "model_1": {"model_0": 1, "model_2": -1},
                     "model_2": {"model_0": 2, "model_1": 1}},
        standard_errors={"model_0": 0.1, "model_1": 0.2, "model_2": 0.3}
    )
    
    # Test best_model method
    assert result.best_model() == "model_0"
    
    # Test metric_name property
    assert result.metric_name == "WAIC"
    
    # Test values property
    assert len(result.values) == 3
    assert result.values["model_0"] == 0
    assert result.values["model_1"] == 1
    assert result.values["model_2"] == 2


def test_comparison_result_to_dict():
    """Test the to_dataframe method of ComparisonResult."""
    # Create a mock comparison result
    result = ComparisonResult(
        metric_name="WAIC",
        values={"model_0": 0, "model_1": 1, "model_2": 2},
        standard_errors={"model_0": 0.1, "model_1": 0.2, "model_2": 0.3}
    )
    
    # Convert to dataframe and then dictionary
    result_dict = result.to_dataframe().to_dict('records')
    
    # Verify the dataframe contains expected values
    assert len(result_dict) == 3  # One record per model
    
    # Find the model_0 record
    model_0_record = next(r for r in result_dict if r['model'] == 'model_0')
    assert model_0_record['WAIC'] == 0
    assert model_0_record['WAIC_se'] == 0.1
    
    # Find the model_1 record
    model_1_record = next(r for r in result_dict if r['model'] == 'model_1')
    assert model_1_record['WAIC'] == 1
    assert model_1_record['WAIC_se'] == 0.2 