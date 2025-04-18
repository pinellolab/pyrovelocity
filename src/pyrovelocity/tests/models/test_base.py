"""Tests for the base component classes in PyroVelocity's modular architecture."""

import pytest
import numpy as np
import torch
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import pyro
import pyro.distributions as dist
from anndata import AnnData
from expression import case, tag, tagged_union

from pyrovelocity.models.base import (
    BaseComponent,
    BaseDynamicsModel,
    BasePriorModel,
    BaseLikelihoodModel,
    BaseObservationModel,
    BaseInferenceGuide,
    ComponentError,
    Result,
)
from pyrovelocity.utils.data import DataSplitter, MiniBatchDataset


class TestBaseComponent:
    """Tests for the BaseComponent class."""

    def test_initialization(self):
        """Test that the base component is initialized correctly."""
        class TestComponent(BaseComponent):
            pass
        
        component = TestComponent(name="test_component")
        assert component.name == "test_component"
    
    def test_handle_error(self):
        """Test that _handle_error creates a proper Error result."""
        class TestComponent(BaseComponent):
            pass
        
        component = TestComponent(name="test_component")
        result = component._handle_error(
            operation="test_operation",
            message="Test error message",
            details={"test_key": "test_value"}
        )
        
        assert isinstance(result, Result.Error)
        assert "TestComponent" in result.message
        assert "test_operation" in result.message
        assert "Test error message" in result.message
        assert result.details is not None
        assert "error" in result.details
        assert isinstance(result.details["error"], ComponentError)
        assert result.details["error"].component == "TestComponent"
        assert result.details["error"].operation == "test_operation"
        assert result.details["error"].message == "Test error message"
        assert result.details["error"].details == {"test_key": "test_value"}
    
    def test_validate_inputs(self):
        """Test that validate_inputs returns an Ok result by default."""
        class TestComponent(BaseComponent):
            pass
        
        component = TestComponent(name="test_component")
        result = component.validate_inputs(test_param="test_value")
        
        assert isinstance(result, Result.Ok)
        assert result.value == {"test_param": "test_value"}


# Concrete implementations for testing
class TestDynamicsModel(BaseDynamicsModel):
    """Concrete implementation of BaseDynamicsModel for testing."""
    
    def _forward_impl(self, time_points, params):
        return np.ones((time_points.shape[0], 10)) * params.get("factor", 1.0)
    
    def _predict_future_states_impl(self, current_state, time_delta, params):
        return current_state * params.get("factor", 1.0)


class TestPriorModel(BasePriorModel):
    """Concrete implementation of BasePriorModel for testing."""
    
    def _register_priors_impl(self, prefix=""):
        with pyro.plate(f"{prefix}gene_plate", 10):
            pyro.sample(f"{prefix}alpha", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
            pyro.sample(f"{prefix}beta", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
    
    def _sample_parameters_impl(self, prefix=""):
        return {
            f"{prefix}alpha": torch.ones(10),
            f"{prefix}beta": torch.ones(10) * 2,
        }


class TestLikelihoodModel(BaseLikelihoodModel):
    """Concrete implementation of BaseLikelihoodModel for testing."""
    
    def _log_prob_impl(self, observations, predictions, scale_factors=None):
        # Simple Poisson log probability
        rate = predictions
        if scale_factors is not None:
            rate = rate * scale_factors.reshape(-1, 1)
        
        # Sum log probabilities across genes
        return dist.Poisson(rate).log_prob(observations).sum(axis=1)
    
    def _sample_impl(self, predictions, scale_factors=None):
        rate = predictions
        if scale_factors is not None:
            rate = rate * scale_factors.reshape(-1, 1)
        
        return np.random.poisson(rate)


class TestObservationModel(BaseObservationModel):
    """Concrete implementation of BaseObservationModel for testing."""
    
    def _prepare_data_impl(self, adata, **kwargs):
        # Mock implementation
        splitter = MagicMock(spec=DataSplitter)
        metadata = {"num_genes": adata.n_vars, "num_cells": adata.n_obs}
        return splitter, metadata
    
    def _create_dataloaders_impl(self, data_splitter, batch_size, **kwargs):
        # Mock implementation
        train_loader = MagicMock(spec=MiniBatchDataset)
        val_loader = MagicMock(spec=MiniBatchDataset)
        return {"train": train_loader, "val": val_loader}
    
    def _preprocess_batch_impl(self, batch):
        # Simple pass-through implementation
        return batch


class TestInferenceGuide(BaseInferenceGuide):
    """Concrete implementation of BaseInferenceGuide for testing."""
    
    def __init__(self, name="inference_guide"):
        super().__init__(name=name)
        self.guide = None
    
    def _setup_guide_impl(self, model, **kwargs):
        # Mock implementation
        self.guide = MagicMock()
    
    def _sample_posterior_impl(self, num_samples=1, **kwargs):
        # Mock implementation
        return {
            "alpha": np.ones((num_samples, 10)),
            "beta": np.ones((num_samples, 10)) * 2,
        }


class TestDynamicsModelImplementation:
    """Tests for the BaseDynamicsModel implementation."""
    
    def test_initialization(self):
        """Test that the dynamics model is initialized correctly."""
        model = TestDynamicsModel(name="test_dynamics")
        assert model.name == "test_dynamics"
    
    def test_forward(self):
        """Test the forward method."""
        model = TestDynamicsModel()
        time_points = np.array([0.0, 1.0, 2.0])
        params = {"factor": 2.0}
        
        result = model.forward(time_points, params)
        
        assert result.shape == (3, 10)
        assert np.all(result == 2.0)
    
    def test_forward_with_validation_error(self):
        """Test the forward method with a validation error."""
        model = TestDynamicsModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error(message="Validation error", details=None)
        
        model.validate_inputs = mock_validate
        
        time_points = np.array([0.0, 1.0, 2.0])
        params = {"factor": 2.0}
        
        with pytest.raises(ValueError, match="Error in dynamics model forward pass"):
            model.forward(time_points, params)
    
    def test_predict_future_states(self):
        """Test the predict_future_states method."""
        model = TestDynamicsModel()
        current_state = np.ones((3, 10))
        time_delta = np.array([1.0, 2.0, 3.0])
        params = {"factor": 2.0}
        
        result = model.predict_future_states(current_state, time_delta, params)
        
        assert result.shape == (3, 10)
        assert np.all(result == 2.0)
    
    def test_predict_future_states_with_validation_error(self):
        """Test the predict_future_states method with a validation error."""
        model = TestDynamicsModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error(message="Validation error", details=None)
        
        model.validate_inputs = mock_validate
        
        current_state = np.ones((3, 10))
        time_delta = np.array([1.0, 2.0, 3.0])
        params = {"factor": 2.0}
        
        with pytest.raises(ValueError, match="Error in dynamics model prediction"):
            model.predict_future_states(current_state, time_delta, params)


class TestPriorModelImplementation:
    """Tests for the BasePriorModel implementation."""
    
    def test_initialization(self):
        """Test that the prior model is initialized correctly."""
        model = TestPriorModel(name="test_prior")
        assert model.name == "test_prior"
    
    def test_register_priors(self):
        """Test the register_priors method."""
        model = TestPriorModel()
        
        # Mock pyro.plate and pyro.sample to avoid actual registration
        with patch("pyro.plate") as mock_plate, patch("pyro.sample") as mock_sample:
            mock_plate.return_value.__enter__.return_value = None
            mock_sample.return_value = None
            
            model.register_priors(prefix="test_")
            
            # Check that pyro.plate was called with the correct arguments
            mock_plate.assert_called_once_with("test_gene_plate", 10)
            
            # Check that pyro.sample was called for each parameter
            assert mock_sample.call_count == 2
            mock_sample.assert_any_call("test_alpha", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
            mock_sample.assert_any_call("test_beta", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
    
    def test_register_priors_with_error(self):
        """Test the register_priors method with an error."""
        model = TestPriorModel()
        
        # Mock _register_priors_impl to raise an exception
        def mock_register_priors_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        model._register_priors_impl = mock_register_priors_impl
        
        with pytest.raises(RuntimeError, match="Test error"):
            model.register_priors()
    
    def test_sample_parameters(self):
        """Test the sample_parameters method."""
        model = TestPriorModel()
        
        params = model.sample_parameters(prefix="test_")
        
        assert "test_alpha" in params
        assert "test_beta" in params
        assert torch.all(params["test_alpha"] == 1.0)
        assert torch.all(params["test_beta"] == 2.0)
    
    def test_sample_parameters_with_error(self):
        """Test the sample_parameters method with an error."""
        model = TestPriorModel()
        
        # Mock _sample_parameters_impl to raise an exception
        def mock_sample_parameters_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        model._sample_parameters_impl = mock_sample_parameters_impl
        
        with pytest.raises(ValueError, match="Failed to sample parameters"):
            model.sample_parameters()


class TestLikelihoodModelImplementation:
    """Tests for the BaseLikelihoodModel implementation."""
    
    def test_initialization(self):
        """Test that the likelihood model is initialized correctly."""
        model = TestLikelihoodModel(name="test_likelihood")
        assert model.name == "test_likelihood"
    
    def test_log_prob(self):
        """Test the log_prob method."""
        model = TestLikelihoodModel()
        observations = np.ones((3, 10))
        predictions = np.ones((3, 10)) * 2.0
        
        result = model.log_prob(observations, predictions)
        
        assert result.shape == (3,)
        
        # Verify log probability calculation (Poisson with rate=2, obs=1)
        expected = dist.Poisson(torch.tensor(2.0)).log_prob(torch.tensor(1.0)).item() * 10
        assert np.allclose(result, expected)
    
    def test_log_prob_with_scale_factors(self):
        """Test the log_prob method with scale factors."""
        model = TestLikelihoodModel()
        observations = np.ones((3, 10))
        predictions = np.ones((3, 10))
        scale_factors = np.array([1.0, 2.0, 3.0])
        
        result = model.log_prob(observations, predictions, scale_factors)
        
        assert result.shape == (3,)
        
        # Verify log probability calculation with scale factors
        for i, sf in enumerate(scale_factors):
            expected = dist.Poisson(torch.tensor(sf)).log_prob(torch.tensor(1.0)).item() * 10
            assert np.isclose(result[i], expected)
    
    def test_log_prob_with_validation_error(self):
        """Test the log_prob method with a validation error."""
        model = TestLikelihoodModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error(message="Validation error", details=None)
        
        model.validate_inputs = mock_validate
        
        observations = np.ones((3, 10))
        predictions = np.ones((3, 10))
        
        with pytest.raises(ValueError, match="Error in likelihood model log_prob"):
            model.log_prob(observations, predictions)
    
    def test_sample(self):
        """Test the sample method."""
        model = TestLikelihoodModel()
        predictions = np.ones((3, 10)) * 5.0
        
        # Mock numpy.random.poisson to return deterministic values
        with patch("numpy.random.poisson") as mock_poisson:
            mock_poisson.return_value = np.ones((3, 10)) * 5.0
            
            result = model.sample(predictions)
            
            assert result.shape == (3, 10)
            assert np.all(result == 5.0)
            mock_poisson.assert_called_once()
    
    def test_sample_with_scale_factors(self):
        """Test the sample method with scale factors."""
        model = TestLikelihoodModel()
        predictions = np.ones((3, 10))
        scale_factors = np.array([1.0, 2.0, 3.0])
        
        # Mock numpy.random.poisson to return deterministic values
        with patch("numpy.random.poisson") as mock_poisson:
            expected_rates = np.array([
                [1.0] * 10,
                [2.0] * 10,
                [3.0] * 10,
            ])
            mock_poisson.return_value = expected_rates
            
            result = model.sample(predictions, scale_factors)
            
            assert result.shape == (3, 10)
            assert np.all(result == expected_rates)
            mock_poisson.assert_called_once()
            # Check that the rates were correctly scaled
            np.testing.assert_array_equal(mock_poisson.call_args[0][0], expected_rates)
    
    def test_sample_with_validation_error(self):
        """Test the sample method with a validation error."""
        model = TestLikelihoodModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error(message="Validation error", details=None)
        
        model.validate_inputs = mock_validate
        
        predictions = np.ones((3, 10))
        
        with pytest.raises(ValueError, match="Error in likelihood model sample"):
            model.sample(predictions)


class TestObservationModelImplementation:
    """Tests for the BaseObservationModel implementation."""
    
    def test_initialization(self):
        """Test that the observation model is initialized correctly."""
        model = TestObservationModel(name="test_observation")
        assert model.name == "test_observation"
    
    def test_prepare_data(self):
        """Test the prepare_data method."""
        model = TestObservationModel()
        
        # Create a mock AnnData object
        adata = MagicMock(spec=AnnData)
        adata.n_vars = 10
        adata.n_obs = 100
        
        splitter, metadata = model.prepare_data(adata)
        
        assert isinstance(splitter, MagicMock)
        assert metadata["num_genes"] == 10
        assert metadata["num_cells"] == 100
    
    def test_prepare_data_with_error(self):
        """Test the prepare_data method with an error."""
        model = TestObservationModel()
        
        # Mock _prepare_data_impl to raise an exception
        def mock_prepare_data_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        model._prepare_data_impl = mock_prepare_data_impl
        
        # Create a mock AnnData object
        adata = MagicMock(spec=AnnData)
        
        with pytest.raises(ValueError, match="Failed to prepare data"):
            model.prepare_data(adata)
    
    def test_create_dataloaders(self):
        """Test the create_dataloaders method."""
        model = TestObservationModel()
        
        # Create a mock DataSplitter
        data_splitter = MagicMock(spec=DataSplitter)
        
        dataloaders = model.create_dataloaders(data_splitter, batch_size=32)
        
        assert "train" in dataloaders
        assert "val" in dataloaders
        assert isinstance(dataloaders["train"], MagicMock)
        assert isinstance(dataloaders["val"], MagicMock)
    
    def test_create_dataloaders_with_error(self):
        """Test the create_dataloaders method with an error."""
        model = TestObservationModel()
        
        # Mock _create_dataloaders_impl to raise an exception
        def mock_create_dataloaders_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        model._create_dataloaders_impl = mock_create_dataloaders_impl
        
        # Create a mock DataSplitter
        data_splitter = MagicMock(spec=DataSplitter)
        
        with pytest.raises(ValueError, match="Failed to create dataloaders"):
            model.create_dataloaders(data_splitter, batch_size=32)
    
    def test_preprocess_batch(self):
        """Test the preprocess_batch method."""
        model = TestObservationModel()
        
        batch = {"x": torch.ones(10), "y": torch.zeros(10)}
        
        result = model.preprocess_batch(batch)
        
        assert result == batch
    
    def test_preprocess_batch_with_error(self):
        """Test the preprocess_batch method with an error."""
        model = TestObservationModel()
        
        # Mock _preprocess_batch_impl to raise an exception
        def mock_preprocess_batch_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        model._preprocess_batch_impl = mock_preprocess_batch_impl
        
        batch = {"x": torch.ones(10), "y": torch.zeros(10)}
        
        with pytest.raises(ValueError, match="Failed to preprocess batch"):
            model.preprocess_batch(batch)


class TestInferenceGuideImplementation:
    """Tests for the BaseInferenceGuide implementation."""
    
    def test_initialization(self):
        """Test that the inference guide is initialized correctly."""
        guide = TestInferenceGuide(name="test_guide")
        assert guide.name == "test_guide"
    
    def test_setup_guide(self):
        """Test the setup_guide method."""
        guide = TestInferenceGuide()
        
        # Create a mock model
        model = MagicMock()
        
        guide.setup_guide(model)
        
        assert guide.guide is not None
    
    def test_setup_guide_with_error(self):
        """Test the setup_guide method with an error."""
        guide = TestInferenceGuide()
        
        # Mock _setup_guide_impl to raise an exception
        def mock_setup_guide_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        guide._setup_guide_impl = mock_setup_guide_impl
        
        # Create a mock model
        model = MagicMock()
        
        with pytest.raises(ValueError, match="Failed to set up guide"):
            guide.setup_guide(model)
    
    def test_sample_posterior(self):
        """Test the sample_posterior method."""
        guide = TestInferenceGuide()
        
        samples = guide.sample_posterior(num_samples=5)
        
        assert "alpha" in samples
        assert "beta" in samples
        assert samples["alpha"].shape == (5, 10)
        assert samples["beta"].shape == (5, 10)
        assert np.all(samples["alpha"] == 1.0)
        assert np.all(samples["beta"] == 2.0)
    
    def test_sample_posterior_with_error(self):
        """Test the sample_posterior method with an error."""
        guide = TestInferenceGuide()
        
        # Mock _sample_posterior_impl to raise an exception
        def mock_sample_posterior_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        guide._sample_posterior_impl = mock_sample_posterior_impl
        
        with pytest.raises(ValueError, match="Failed to sample from posterior"):
            guide.sample_posterior()