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
from expression import Result, case, tag, tagged_union

from pyrovelocity.models.components.base import (
    BaseComponent,
    BaseDynamicsModel,
    BasePriorModel,
    BaseLikelihoodModel,
    BaseObservationModel,
    BaseInferenceGuide,
    ComponentError,
)
# Removed import of non-existent module


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
        
        assert result.is_error()
        assert "TestComponent" in result.error
        assert "test_operation" in result.error
        assert "Test error message" in result.error
    
    def test_validate_inputs(self):
        """Test that validate_inputs returns an Ok result by default."""
        class TestComponent(BaseComponent):
            pass
        
        component = TestComponent(name="test_component")
        result = component.validate_inputs(test_param="test_value")
        
        assert result.is_ok()
        assert result.ok == {"test_param": "test_value"}


# Concrete implementations for testing
class TestDynamicsModel(BaseDynamicsModel):
    """Concrete implementation of BaseDynamicsModel for testing."""
    
    def _forward_impl(
        self,
        u,
        s,
        alpha,
        beta,
        gamma,
        scaling=None,
        t=None,
    ):
        # Simple implementation for testing
        batch_size = u.shape[0]
        gene_count = alpha.shape[0]
        
        # Create mock predictions based on parameters
        u_pred = torch.ones((batch_size, gene_count)) * alpha.mean()
        s_pred = torch.ones((batch_size, gene_count)) * beta.mean()
        
        return u_pred, s_pred
    
    def _predict_future_states_impl(
        self,
        current_state,
        time_delta,
        alpha,
        beta,
        gamma,
        scaling=None,
    ):
        # Simple implementation for testing
        u_current, s_current = current_state
        batch_size = u_current.shape[0]
        gene_count = alpha.shape[0]
        
        # Create mock future states based on current state and parameters
        u_future = u_current * (1 + time_delta.reshape(-1, 1) * 0.1)
        s_future = s_current * (1 + time_delta.reshape(-1, 1) * 0.2)
        
        return u_future, s_future


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


# Create a wrapper for BaseLikelihoodModel that handles PyTorch tensors properly
class PyTorchLikelihoodModel(BaseLikelihoodModel):
    """Base class for likelihood models that work with PyTorch tensors."""
    
    def log_prob(self, observations, predictions, scale_factors=None):
        """
        Calculate log probability of observations given predictions.
        
        This method overrides the base class method to handle PyTorch tensors.
        
        Args:
            observations: Observed gene expression
            predictions: Predicted gene expression from dynamics model
            scale_factors: Optional scaling factors for observations
            
        Returns:
            Log probability of observations
        """
        # Validate inputs
        validation_result = self.validate_inputs(
            observations=observations,
            predictions=predictions,
            scale_factors=scale_factors,
        )
        
        if validation_result.is_error():
            raise ValueError(f"Error in likelihood model log_prob: {validation_result.error}")
        
        # Call implementation
        return self._log_prob_impl(observations, predictions, scale_factors)
    
    def sample(self, predictions, scale_factors=None):
        """
        Sample observations from the likelihood model.
        
        This method overrides the base class method to handle PyTorch tensors.
        
        Args:
            predictions: Predicted gene expression from dynamics model
            scale_factors: Optional scaling factors for observations
            
        Returns:
            Sampled observations
        """
        # Validate inputs
        validation_result = self.validate_inputs(
            predictions=predictions,
            scale_factors=scale_factors,
        )
        
        if validation_result.is_error():
            raise ValueError(f"Error in likelihood model sample: {validation_result.error}")
        
        # Call implementation
        return self._sample_impl(predictions, scale_factors)


class TestLikelihoodModel(PyTorchLikelihoodModel):
    """Concrete implementation of BaseLikelihoodModel for testing."""
    
    def _log_prob_impl(self, observations, predictions, scale_factors=None):
        # Convert numpy arrays to torch tensors if needed
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations, dtype=torch.float32)
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions, dtype=torch.float32)
        if scale_factors is not None and isinstance(scale_factors, np.ndarray):
            scale_factors = torch.tensor(scale_factors, dtype=torch.float32)
        
        # Simple Poisson log probability
        rate = predictions
        if scale_factors is not None:
            rate = rate * scale_factors.reshape(-1, 1)
        
        # Sum log probabilities across genes
        return dist.Poisson(rate).log_prob(observations).sum(dim=1)
    
    def _sample_impl(self, predictions, scale_factors=None):
        # Convert numpy arrays to torch tensors if needed
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions, dtype=torch.float32)
        if scale_factors is not None and isinstance(scale_factors, np.ndarray):
            scale_factors = torch.tensor(scale_factors, dtype=torch.float32)
        
        rate = predictions
        if scale_factors is not None:
            rate = rate * scale_factors.reshape(-1, 1)
        
        # Use torch's Poisson distribution to sample
        samples = torch.poisson(rate)
        return samples


class TestObservationModel(BaseObservationModel):
    """Concrete implementation of BaseObservationModel for testing."""
    
    def _prepare_data_impl(self, adata, **kwargs):
        # Mock implementation
        # Return a dictionary with tensor values to match the type annotation
        return {
            "X": torch.ones(adata.n_obs, adata.n_vars),
            "metadata": torch.tensor([adata.n_vars, adata.n_obs])
        }
    
    def _create_dataloaders_impl(self, data, batch_size, **kwargs):
        # Create actual DataLoader objects to match the type annotation
        # We'll use a simple TensorDataset with the data
        dataset = torch.utils.data.TensorDataset(data["X"])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
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
        
        # Create test inputs
        batch_size = 3
        gene_count = 10
        u = torch.ones((batch_size, gene_count))
        s = torch.ones((batch_size, gene_count)) * 0.5
        alpha = torch.ones(gene_count) * 2.0
        beta = torch.ones(gene_count) * 1.5
        gamma = torch.ones(gene_count) * 0.8
        
        # Call forward method
        u_pred, s_pred = model.forward(u, s, alpha, beta, gamma)
        
        # Check results
        assert u_pred.shape == (batch_size, gene_count)
        assert s_pred.shape == (batch_size, gene_count)
        assert torch.all(u_pred > 0)
        assert torch.all(s_pred > 0)
    
    def test_forward_with_validation_error(self):
        """Test the forward method with a validation error."""
        model = TestDynamicsModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error("Validation error")
        
        model.validate_inputs = mock_validate
        
        # Create test inputs
        batch_size = 3
        gene_count = 10
        u = torch.ones((batch_size, gene_count))
        s = torch.ones((batch_size, gene_count)) * 0.5
        alpha = torch.ones(gene_count) * 2.0
        beta = torch.ones(gene_count) * 1.5
        gamma = torch.ones(gene_count) * 0.8
        
        with pytest.raises(ValueError, match="Error in dynamics model forward pass"):
            model.forward(u, s, alpha, beta, gamma)
    
    def test_predict_future_states(self):
        """Test the predict_future_states method."""
        model = TestDynamicsModel()
        
        # Create test inputs
        batch_size = 3
        gene_count = 10
        u_current = torch.ones((batch_size, gene_count))
        s_current = torch.ones((batch_size, gene_count)) * 0.5
        current_state = (u_current, s_current)
        time_delta = torch.tensor([1.0, 2.0, 3.0])
        alpha = torch.ones(gene_count) * 2.0
        beta = torch.ones(gene_count) * 1.5
        gamma = torch.ones(gene_count) * 0.8
        
        # Call predict_future_states method
        u_future, s_future = model.predict_future_states(
            current_state, time_delta, alpha, beta, gamma
        )
        
        # Check results
        assert u_future.shape == (batch_size, gene_count)
        assert s_future.shape == (batch_size, gene_count)
        # Check that future values are different from current values
        assert not torch.allclose(u_future, u_current)
        assert not torch.allclose(s_future, s_current)
    
    def test_predict_future_states_with_validation_error(self):
        """Test the predict_future_states method with a validation error."""
        model = TestDynamicsModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error("Validation error")
        
        model.validate_inputs = mock_validate
        
        # Create test inputs
        batch_size = 3
        gene_count = 10
        u_current = torch.ones((batch_size, gene_count))
        s_current = torch.ones((batch_size, gene_count)) * 0.5
        current_state = (u_current, s_current)
        time_delta = torch.tensor([1.0, 2.0, 3.0])
        alpha = torch.ones(gene_count) * 2.0
        beta = torch.ones(gene_count) * 1.5
        gamma = torch.ones(gene_count) * 0.8
        
        with pytest.raises(ValueError, match="Error in dynamics model prediction"):
            model.predict_future_states(current_state, time_delta, alpha, beta, gamma)


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
            
            # Check the calls manually to avoid issues with object equality
            call_args_list = mock_sample.call_args_list
            param_names = [call_args[0][0] for call_args in call_args_list]
            
            # Check that both parameters were registered
            assert "test_alpha" in param_names
            assert "test_beta" in param_names
            
            # Check that the distributions are LogNormal with correct parameters
            for call_args in call_args_list:
                name, distribution = call_args[0]
                assert isinstance(distribution, dist.LogNormal)
                assert torch.allclose(distribution.loc, torch.tensor(0.0))
                assert torch.allclose(distribution.scale, torch.tensor(1.0))
    
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
        observations = torch.ones((3, 10))
        predictions = torch.ones((3, 10)) * 2.0
        
        result = model.log_prob(observations, predictions)
        
        assert result.shape == (3,)
        
        # Verify log probability calculation (Poisson with rate=2, obs=1)
        expected = dist.Poisson(torch.tensor(2.0)).log_prob(torch.tensor(1.0)).item() * 10
        assert torch.allclose(result, torch.tensor([expected, expected, expected]))
    
    def test_log_prob_with_scale_factors(self):
        """Test the log_prob method with scale factors."""
        model = TestLikelihoodModel()
        observations = torch.ones((3, 10))
        predictions = torch.ones((3, 10))
        scale_factors = torch.tensor([1.0, 2.0, 3.0])
        
        result = model.log_prob(observations, predictions, scale_factors)
        
        assert result.shape == (3,)
        
        # Verify log probability calculation with scale factors
        for i, sf in enumerate(scale_factors):
            expected = dist.Poisson(sf).log_prob(torch.tensor(1.0)).item() * 10
            assert torch.isclose(result[i], torch.tensor(expected))
    
    def test_log_prob_with_validation_error(self):
        """Test the log_prob method with a validation error."""
        model = TestLikelihoodModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error("Validation error")
        
        model.validate_inputs = mock_validate
        
        observations = torch.ones((3, 10))
        predictions = torch.ones((3, 10))
        
        with pytest.raises(ValueError, match="Error in likelihood model log_prob"):
            model.log_prob(observations, predictions)
    
    def test_sample(self):
        """Test the sample method."""
        model = TestLikelihoodModel()
        predictions = torch.ones((3, 10)) * 5.0
        
        # Mock torch.poisson to return deterministic values
        with patch("torch.poisson") as mock_poisson:
            mock_poisson.return_value = torch.ones((3, 10)) * 5.0
            
            result = model.sample(predictions)
            
            assert result.shape == (3, 10)
            assert torch.all(result == 5.0)
            mock_poisson.assert_called_once()
    
    def test_sample_with_scale_factors(self):
        """Test the sample method with scale factors."""
        model = TestLikelihoodModel()
        predictions = torch.ones((3, 10))
        scale_factors = torch.tensor([1.0, 2.0, 3.0])
        
        # Create expected rates tensor
        expected_rates = torch.tensor([
            [1.0] * 10,
            [2.0] * 10,
            [3.0] * 10,
        ])
        
        # Mock torch.poisson to return deterministic values
        with patch("torch.poisson") as mock_poisson:
            mock_poisson.return_value = expected_rates
            
            result = model.sample(predictions, scale_factors)
            
            assert result.shape == (3, 10)
            assert torch.all(result == expected_rates)
            mock_poisson.assert_called_once()
            # Check that the rates were correctly scaled
            assert torch.allclose(mock_poisson.call_args[0][0], expected_rates)
    
    def test_sample_with_validation_error(self):
        """Test the sample method with a validation error."""
        model = TestLikelihoodModel()
        
        # Override validate_inputs to return an error
        def mock_validate(*args, **kwargs):
            return Result.Error("Validation error")
        
        model.validate_inputs = mock_validate
        
        predictions = torch.ones((3, 10))
        
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
        
        result = model.prepare_data(adata)
        
        assert isinstance(result, dict)
        assert "X" in result
        assert "metadata" in result
        assert isinstance(result["X"], torch.Tensor)
        assert isinstance(result["metadata"], torch.Tensor)
        assert result["X"].shape == (100, 10)
        assert torch.equal(result["metadata"], torch.tensor([10, 100]))
    
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
        
        # Create a data dictionary with tensors
        data = {"X": torch.ones(100, 10)}
        
        dataloaders = model.create_dataloaders(data, batch_size=32)
        
        assert "train" in dataloaders
        assert "val" in dataloaders
        assert isinstance(dataloaders["train"], torch.utils.data.DataLoader)
        assert isinstance(dataloaders["val"], torch.utils.data.DataLoader)
    
    def test_create_dataloaders_with_error(self):
        """Test the create_dataloaders method with an error."""
        model = TestObservationModel()
        
        # Mock _create_dataloaders_impl to raise an exception
        def mock_create_dataloaders_impl(*args, **kwargs):
            raise RuntimeError("Test error")
        
        model._create_dataloaders_impl = mock_create_dataloaders_impl
        
        # Create a data dictionary with tensors
        data = {"X": torch.ones(100, 10)}
        
        with pytest.raises(ValueError, match="Failed to create dataloaders"):
            model.create_dataloaders(data, batch_size=32)
    
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