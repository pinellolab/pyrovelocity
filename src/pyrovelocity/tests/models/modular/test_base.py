"""
Tests for the base component classes in PyroVelocity's modular architecture.

This module contains tests for the abstract base classes defined in components/base.py,
verifying that they correctly implement the Protocol interfaces and provide the
expected functionality.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pyro
import pytest
import torch
import torch.utils.data
from anndata import AnnData
from jaxtyping import Array, Float

from pyrovelocity.models.modular.components.base import (
    BaseComponent,
    BaseDynamicsModel,
    BaseInferenceGuide,
    BaseLikelihoodModel,
    BaseObservationModel,
    BasePriorModel,
    ComponentError,
    PyroBufferMixin,
)
from pyrovelocity.models.modular.interfaces import (
    BatchTensor,
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ModelState,
    ObservationModel,
    ParamTensor,
    PriorModel,
)


class TestBaseComponent:
    """Tests for the BaseComponent class."""

    def test_initialization(self):
        """Test that BaseComponent initializes correctly."""
        component = BaseComponent(name="test_component")
        assert component.name == "test_component"

    def test_handle_error(self):
        """Test that _handle_error creates a Result.Error with ComponentError."""
        component = BaseComponent(name="test_component")
        result = component._handle_error(
            operation="test_operation",
            message="test message",
            details={"key": "value"},
        )

        assert result.is_error()
        assert "BaseComponent.test_operation: test message" in result.error


class TestComponentError:
    """Tests for the ComponentError class."""

    def test_initialization(self):
        """Test that ComponentError initializes correctly."""
        error = ComponentError(
            component="TestComponent",
            operation="test_operation",
            message="test message",
            details={"key": "value"},
        )

        assert error.component == "TestComponent"
        assert error.operation == "test_operation"
        assert error.message == "test message"
        assert error.details == {"key": "value"}

    def test_initialization_with_default_details(self):
        """Test that ComponentError initializes with default details."""
        error = ComponentError(
            component="TestComponent",
            operation="test_operation",
            message="test message",
        )

        assert error.details == {}


class TestPyroBufferMixin:
    """Tests for the PyroBufferMixin class."""

    def test_register_buffer(self):
        """Test that register_buffer sets an attribute."""
        class TestClass(PyroBufferMixin):
            pass

        obj = TestClass()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        obj.register_buffer("test_buffer", tensor)

        assert hasattr(obj, "test_buffer")
        assert torch.all(obj.test_buffer == tensor)


class TestBaseDynamicsModel:
    """Tests for the BaseDynamicsModel class."""

    class ConcreteDynamicsModel(BaseDynamicsModel):
        """Concrete implementation of BaseDynamicsModel for testing."""

        def _forward_impl(
            self,
            u: BatchTensor,
            s: BatchTensor,
            alpha: ParamTensor,
            beta: ParamTensor,
            gamma: ParamTensor,
            scaling: Optional[ParamTensor] = None,
            t: Optional[BatchTensor] = None,
            **kwargs: Any,
        ) -> Tuple[BatchTensor, BatchTensor]:
            """Implement abstract method."""
            return u, s

        def _predict_future_states_impl(
            self,
            current_state: Tuple[BatchTensor, BatchTensor],
            time_delta: BatchTensor,
            alpha: ParamTensor,
            beta: ParamTensor,
            gamma: ParamTensor,
            scaling: Optional[ParamTensor] = None,
        ) -> Tuple[BatchTensor, BatchTensor]:
            """Implement abstract method."""
            u, s = current_state
            return u, s

        def _steady_state_impl(
            self,
            alpha: Union[ParamTensor, Array],
            beta: Union[ParamTensor, Array],
            gamma: Union[ParamTensor, Array],
            **kwargs: Any,
        ) -> Tuple[Union[ParamTensor, Array], Union[ParamTensor, Array]]:
            """Implement abstract method.

            Args:
                alpha: Transcription rates [genes]
                beta: Splicing rates [genes]
                gamma: Degradation rates [genes]
                **kwargs: Additional model-specific parameters

            Returns:
                Tuple of (unspliced_steady_state, spliced_steady_state)
                each with shape [genes]
            """
            # At steady state, du/dt = 0 and ds/dt = 0
            # From du/dt = 0: alpha - beta * u = 0 => u = alpha / beta
            u_ss = alpha / beta

            # From ds/dt = 0: beta * u - gamma * s = 0 => s = beta * u / gamma
            # Substituting u = alpha / beta: s = beta * (alpha / beta) / gamma = alpha / gamma
            s_ss = alpha / gamma

            return u_ss, s_ss

    def test_initialization(self):
        """Test that BaseDynamicsModel initializes correctly."""
        model = self.ConcreteDynamicsModel(
            name="test_dynamics",
            shared_time=True,
            t_scale_on=False,
            cell_specific_kinetics="test",
            kinetics_num=10,
            extra_param="value",
        )

        assert model.name == "test_dynamics"
        assert model.shared_time is True
        assert model.t_scale_on is False
        assert model.cell_specific_kinetics == "test"
        assert model.kinetics_num == 10
        assert hasattr(model, "extra_param")
        assert model.extra_param == "value"

    def test_forward(self):
        """Test that forward calls _forward_impl with validated inputs."""
        model = self.ConcreteDynamicsModel()

        # Create test inputs
        u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        s = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        alpha = torch.tensor([0.1, 0.2])
        beta = torch.tensor([0.3, 0.4])
        gamma = torch.tensor([0.5, 0.6])

        # Mock validate_inputs to return a successful result
        model.validate_inputs = MagicMock(return_value=pytest.importorskip("expression").Result.Ok({}))

        # Mock _forward_impl
        model._forward_impl = MagicMock(return_value=(u, s))

        # Call forward
        result = model.forward(u, s, alpha, beta, gamma)

        # Check that validate_inputs was called with the correct arguments
        model.validate_inputs.assert_called_once()

        # Check that _forward_impl was called with the correct arguments
        model._forward_impl.assert_called_once_with(u, s, alpha, beta, gamma, None, None)

        # Check that the result is correct
        assert result == (u, s)

    def test_predict_future_states(self):
        """Test that predict_future_states calls _predict_future_states_impl with validated inputs."""
        model = self.ConcreteDynamicsModel()

        # Create test inputs
        u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        s = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        current_state = (u, s)
        time_delta = torch.tensor([0.1, 0.2])
        alpha = torch.tensor([0.1, 0.2])
        beta = torch.tensor([0.3, 0.4])
        gamma = torch.tensor([0.5, 0.6])

        # Mock validate_inputs to return a successful result
        model.validate_inputs = MagicMock(return_value=pytest.importorskip("expression").Result.Ok({}))

        # Mock _predict_future_states_impl
        model._predict_future_states_impl = MagicMock(return_value=(u, s))

        # Call predict_future_states
        result = model.predict_future_states(current_state, time_delta, alpha, beta, gamma)

        # Check that validate_inputs was called with the correct arguments
        model.validate_inputs.assert_called_once()

        # Check that _predict_future_states_impl was called with the correct arguments
        model._predict_future_states_impl.assert_called_once_with(
            current_state, time_delta, alpha, beta, gamma, None
        )

        # Check that the result is correct
        assert result == (u, s)

    def test_steady_state(self):
        """Test that steady_state returns the correct values."""
        model = self.ConcreteDynamicsModel()

        # Create test inputs
        alpha = torch.tensor([1.0, 2.0])
        beta = torch.tensor([2.0, 4.0])
        gamma = torch.tensor([5.0, 10.0])

        # Call steady_state
        u_ss, s_ss = model.steady_state(alpha, beta, gamma)

        # Check that the result is correct
        assert torch.allclose(u_ss, alpha / beta)
        assert torch.allclose(s_ss, alpha / gamma)


class TestBasePriorModel:
    """Tests for the BasePriorModel class."""

    class ConcretePriorModel(BasePriorModel):
        """Concrete implementation of BasePriorModel for testing."""

        def _register_priors_impl(self, prefix: str = "") -> None:
            """Implement abstract method."""
            pass

        def _sample_parameters_impl(self, prefix: str = "") -> Dict[str, Any]:
            """Implement abstract method."""
            return {"alpha": 0.1, "beta": 0.2, "gamma": 0.3}

        def forward(
            self,
            u_obs: BatchTensor,
            s_obs: BatchTensor,
            plate: pyro.plate,
            **kwargs: Any,
        ) -> ModelState:
            """Implement forward method."""
            return {"alpha": 0.1, "beta": 0.2, "gamma": 0.3}

    def test_initialization(self):
        """Test that BasePriorModel initializes correctly."""
        model = self.ConcretePriorModel(name="test_prior")
        assert model.name == "test_prior"

    def test_register_priors(self):
        """Test that register_priors calls _register_priors_impl."""
        model = self.ConcretePriorModel()

        # Mock _register_priors_impl
        model._register_priors_impl = MagicMock()

        # Call register_priors
        model.register_priors(prefix="test_")

        # Check that _register_priors_impl was called with the correct arguments
        model._register_priors_impl.assert_called_once_with("test_")

    def test_sample_parameters(self):
        """Test that sample_parameters calls _sample_parameters_impl."""
        model = self.ConcretePriorModel()

        # Mock _sample_parameters_impl
        expected_result = {"alpha": 0.1, "beta": 0.2, "gamma": 0.3}
        model._sample_parameters_impl = MagicMock(return_value=expected_result)

        # Call sample_parameters
        result = model.sample_parameters(prefix="test_")

        # Check that _sample_parameters_impl was called with the correct arguments
        model._sample_parameters_impl.assert_called_once_with("test_", None)

        # Check that the result is correct
        assert result == expected_result

    def test_sample_parameters_error_handling(self):
        """Test that sample_parameters handles errors correctly."""
        model = self.ConcretePriorModel()

        # Mock _sample_parameters_impl to raise an exception
        model._sample_parameters_impl = MagicMock(side_effect=ValueError("Test error"))

        # Call sample_parameters and check that it raises a ValueError
        with pytest.raises(ValueError, match="Failed to sample parameters"):
            model.sample_parameters()


class TestBaseLikelihoodModel:
    """Tests for the BaseLikelihoodModel class."""

    class ConcreteLikelihoodModel(BaseLikelihoodModel):
        """Concrete implementation of BaseLikelihoodModel for testing."""

        def _log_prob_impl(
            self,
            observations: Float[Array, "batch_size genes"],
            predictions: Float[Array, "batch_size genes"],
            scale_factors: Optional[Float[Array, "batch_size"]] = None,
        ) -> Float[Array, "batch_size"]:
            """Implement abstract method."""
            import jax.numpy as jnp
            return jnp.zeros(observations.shape[0])

        def _sample_impl(
            self,
            predictions: Float[Array, "batch_size genes"],
            scale_factors: Optional[Float[Array, "batch_size"]] = None,
        ) -> Float[Array, "batch_size genes"]:
            """Implement abstract method."""
            import jax.numpy as jnp
            return predictions

        def forward(
            self,
            u_obs: BatchTensor,
            s_obs: BatchTensor,
            u_logits: BatchTensor,
            s_logits: BatchTensor,
            plate: pyro.plate,
            **kwargs: Any,
        ) -> None:
            """Implement forward method."""
            pass

    def test_initialization(self):
        """Test that BaseLikelihoodModel initializes correctly."""
        model = self.ConcreteLikelihoodModel(name="test_likelihood")
        assert model.name == "test_likelihood"

    def test_log_prob(self):
        """Test that log_prob calls _log_prob_impl."""
        model = self.ConcreteLikelihoodModel()

        # Create test inputs
        import jax.numpy as jnp
        observations = jnp.zeros((2, 3))
        predictions = jnp.zeros((2, 3))

        # Mock _log_prob_impl
        expected_result = jnp.zeros(2)
        model._log_prob_impl = MagicMock(return_value=expected_result)

        # Call log_prob
        result = model.log_prob(observations, predictions)

        # Check that _log_prob_impl was called with the correct arguments
        model._log_prob_impl.assert_called_once()

        # Check that the result is correct
        assert result is expected_result

    def test_sample(self):
        """Test that sample calls _sample_impl."""
        model = self.ConcreteLikelihoodModel()

        # Create test inputs
        import jax.numpy as jnp
        predictions = jnp.zeros((2, 3))

        # Mock _sample_impl
        expected_result = jnp.zeros((2, 3))
        model._sample_impl = MagicMock(return_value=expected_result)

        # Call sample
        result = model.sample(predictions)

        # Check that _sample_impl was called with the correct arguments
        model._sample_impl.assert_called_once()

        # Check that the result is correct
        assert result is expected_result


class TestBaseObservationModel:
    """Tests for the BaseObservationModel class."""

    class ConcreteObservationModel(BaseObservationModel):
        """Concrete implementation of BaseObservationModel for testing."""

        def _prepare_data_impl(
            self, adata: AnnData, **kwargs: Any
        ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
            """Implement abstract method."""
            return {"data": torch.zeros(10)}

        def _create_dataloaders_impl(
            self, data: Dict[str, torch.Tensor], **kwargs: Any
        ) -> Dict[str, torch.utils.data.DataLoader]:
            """Implement abstract method."""
            return {"dataloader": torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(data["data"])
            )}

        def _preprocess_batch_impl(
            self, batch: Dict[str, torch.Tensor]
        ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
            """Implement abstract method."""
            return batch

        def forward(
            self,
            u_obs: BatchTensor,
            s_obs: BatchTensor,
            **kwargs: Any,
        ) -> Tuple[BatchTensor, BatchTensor]:
            """Implement forward method."""
            return u_obs, s_obs

    def test_initialization(self):
        """Test that BaseObservationModel initializes correctly."""
        model = self.ConcreteObservationModel(name="test_observation")
        assert model.name == "test_observation"

    def test_prepare_data(self):
        """Test that prepare_data calls _prepare_data_impl."""
        model = self.ConcreteObservationModel()

        # Create a mock AnnData object
        adata = MagicMock(spec=AnnData)

        # Mock _prepare_data_impl
        expected_result = {"data": torch.zeros(10)}
        model._prepare_data_impl = MagicMock(return_value=expected_result)

        # Call prepare_data
        result = model.prepare_data(adata, param="value")

        # Check that _prepare_data_impl was called with the correct arguments
        model._prepare_data_impl.assert_called_once_with(adata, param="value")

        # Check that the result is correct
        assert result == expected_result

    def test_prepare_data_error_handling(self):
        """Test that prepare_data handles errors correctly."""
        model = self.ConcreteObservationModel()

        # Create a mock AnnData object
        adata = MagicMock(spec=AnnData)

        # Mock _prepare_data_impl to raise an exception
        model._prepare_data_impl = MagicMock(side_effect=ValueError("Test error"))

        # Call prepare_data and check that it raises a ValueError
        with pytest.raises(ValueError, match="Failed to prepare data"):
            model.prepare_data(adata)

    def test_create_dataloaders(self):
        """Test that create_dataloaders calls _create_dataloaders_impl."""
        model = self.ConcreteObservationModel()

        # Create test inputs
        data = {"data": torch.zeros(10)}

        # Mock _create_dataloaders_impl
        mock_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(1)))
        expected_result = {"dataloader": mock_dataloader}
        model._create_dataloaders_impl = MagicMock(return_value=expected_result)

        # Call create_dataloaders
        result = model.create_dataloaders(data, param="value")

        # Check that _create_dataloaders_impl was called with the correct arguments
        model._create_dataloaders_impl.assert_called_once_with(data, param="value")

        # Check that the result is correct
        assert result == expected_result

    def test_create_dataloaders_error_handling(self):
        """Test that create_dataloaders handles errors correctly."""
        model = self.ConcreteObservationModel()

        # Create test inputs
        data = {"data": torch.zeros(10)}

        # Mock _create_dataloaders_impl to raise an exception
        model._create_dataloaders_impl = MagicMock(side_effect=ValueError("Test error"))

        # Call create_dataloaders and check that it raises a ValueError
        with pytest.raises(ValueError, match="Failed to create dataloaders"):
            model.create_dataloaders(data)

    def test_preprocess_batch(self):
        """Test that preprocess_batch calls _preprocess_batch_impl."""
        model = self.ConcreteObservationModel()

        # Create test inputs
        batch = {"data": torch.zeros(10)}

        # Mock _preprocess_batch_impl
        expected_result = {"processed_data": torch.zeros(10)}
        model._preprocess_batch_impl = MagicMock(return_value=expected_result)

        # Call preprocess_batch
        result = model.preprocess_batch(batch)

        # Check that _preprocess_batch_impl was called with the correct arguments
        model._preprocess_batch_impl.assert_called_once_with(batch)

        # Check that the result is correct
        assert result == expected_result

    def test_preprocess_batch_error_handling(self):
        """Test that preprocess_batch handles errors correctly."""
        model = self.ConcreteObservationModel()

        # Create test inputs
        batch = {"data": torch.zeros(10)}

        # Mock _preprocess_batch_impl to raise an exception
        model._preprocess_batch_impl = MagicMock(side_effect=ValueError("Test error"))

        # Call preprocess_batch and check that it raises a ValueError
        with pytest.raises(ValueError, match="Failed to preprocess batch"):
            model.preprocess_batch(batch)


class TestBaseInferenceGuide:
    """Tests for the BaseInferenceGuide class."""

    class ConcreteInferenceGuide(BaseInferenceGuide):
        """Concrete implementation of BaseInferenceGuide for testing."""

        def _setup_guide_impl(self, model: Callable, **kwargs) -> None:
            """Implement abstract method."""
            self.model = model
            self.guide = lambda *args, **kwargs: None

        def _sample_posterior_impl(self, **kwargs) -> Dict[str, torch.Tensor]:
            """Implement abstract method."""
            return {"param": torch.zeros(10)}

        def __call__(
            self,
            model: Callable,
            *args: Any,
            **kwargs: Any,
        ) -> Callable:
            """Implement __call__ method."""
            return lambda *args, **kwargs: None

    def test_initialization(self):
        """Test that BaseInferenceGuide initializes correctly."""
        guide = self.ConcreteInferenceGuide(name="test_guide")
        assert guide.name == "test_guide"

    def test_setup_guide(self):
        """Test that setup_guide calls _setup_guide_impl."""
        guide = self.ConcreteInferenceGuide()

        # Create a mock model function
        model = MagicMock()

        # Mock _setup_guide_impl
        guide._setup_guide_impl = MagicMock()

        # Call setup_guide
        guide.setup_guide(model, param="value")

        # Check that _setup_guide_impl was called with the correct arguments
        guide._setup_guide_impl.assert_called_once_with(model, param="value")

    def test_setup_guide_error_handling(self):
        """Test that setup_guide handles errors correctly."""
        guide = self.ConcreteInferenceGuide()

        # Create a mock model function
        model = MagicMock()

        # Mock _setup_guide_impl to raise an exception
        guide._setup_guide_impl = MagicMock(side_effect=ValueError("Test error"))

        # Call setup_guide and check that it raises a ValueError
        with pytest.raises(ValueError, match="Failed to set up guide"):
            guide.setup_guide(model)

    def test_sample_posterior(self):
        """Test that sample_posterior calls _sample_posterior_impl."""
        guide = self.ConcreteInferenceGuide()

        # Mock _sample_posterior_impl
        expected_result = {"param": torch.zeros(10)}
        guide._sample_posterior_impl = MagicMock(return_value=expected_result)

        # Call sample_posterior
        result = guide.sample_posterior(param="value")

        # Check that _sample_posterior_impl was called with the correct arguments
        guide._sample_posterior_impl.assert_called_once_with(param="value")

        # Check that the result is correct
        assert result == expected_result

    def test_sample_posterior_error_handling(self):
        """Test that sample_posterior handles errors correctly."""
        guide = self.ConcreteInferenceGuide()

        # Mock _sample_posterior_impl to raise an exception
        guide._sample_posterior_impl = MagicMock(side_effect=ValueError("Test error"))

        # Call sample_posterior and check that it raises a ValueError
        with pytest.raises(ValueError, match="Failed to sample from posterior"):
            guide.sample_posterior()
