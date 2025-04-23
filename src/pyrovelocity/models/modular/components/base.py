"""
Base component classes for PyroVelocity's modular architecture.

This module provides abstract base classes for each component type that adhere to
the Protocol interfaces defined in `interfaces.py`. These base classes include
common functionality and utilities that would be useful across different
implementations of each component.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

# Remove JAX dependency
import pyro
import torch
import torch.utils.data
from anndata import AnnData
from beartype import beartype
from beartype.typing import Callable
from expression import Result, case, tag, tagged_union

# We're using PyTorch tensors, not JAX arrays
from pyrovelocity.models.modular.interfaces import (
    BatchTensor,
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    ParamTensor,
    PriorModel,
)


class ComponentError:
    """
    Error information for component operations.

    This class represents errors that occur during component operations.
    It includes information about the component, operation, error message,
    and additional details.
    """

    def __init__(
        self,
        component: str,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.component = component
        self.operation = operation
        self.message = message
        self.details = details or {}


class BaseComponent:
    """
    Base class for all components in PyroVelocity's modular architecture.

    This class provides common functionality for all components, including
    error handling and input validation.
    """

    def __init__(self, name: str):
        """
        Initialize the component.

        Args:
            name: A unique name for this component instance.
        """
        self.name = name

    def _handle_error(
        self,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """
        Create an Error result for a component operation.

        Args:
            operation: The name of the operation that failed.
            message: A descriptive error message.
            details: Optional additional error details.

        Returns:
            A Result.Error containing a ComponentError.
        """
        error = ComponentError(
            component=self.__class__.__name__,
            operation=operation,
            message=message,
            details=details or {},
        )

        error_message = f"{self.__class__.__name__}.{operation}: {message}"
        return Result.Error(error_message)

    def validate_inputs(self, **kwargs) -> Result:
        """
        Validate inputs for a component operation.

        This method should be overridden by subclasses to provide specific
        validation logic. The default implementation accepts all inputs.

        Args:
            **kwargs: The inputs to validate.

        Returns:
            A Result.Ok containing the validated inputs, or a Result.Error
            if validation fails.
        """
        return Result.Ok(kwargs)


class BaseDynamicsModel(BaseComponent, DynamicsModel, abc.ABC):
    """
    Base class for dynamics models that define gene expression evolution over time.

    This class implements the DynamicsModel protocol and provides common
    functionality for dynamics models. It supports analytical and numerical solutions
    to the RNA velocity differential equations.
    """

    def __init__(
        self,
        name: str = "dynamics_model",
        shared_time: bool = True,
        t_scale_on: bool = False,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the dynamics model.

        Args:
            name: A unique name for this component instance.
            shared_time: Whether to use shared time across cells.
            t_scale_on: Whether to use time scaling.
            cell_specific_kinetics: Type of cell-specific kinetics.
            kinetics_num: Number of kinetics.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name)

        # Store dynamics parameters
        self.shared_time = shared_time
        self.t_scale_on = t_scale_on
        self.cell_specific_kinetics = cell_specific_kinetics
        self.kinetics_num = kinetics_num

        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the expected unspliced and spliced RNA counts based on the dynamics model.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, and parameters

        Returns:
            Updated context dictionary with expected counts
        """
        # Extract parameters from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")

        # Extract parameters from context or use defaults
        alpha = context.get("alpha", torch.tensor(1.0))
        beta = context.get("beta", torch.tensor(1.0))
        gamma = context.get("gamma", torch.tensor(1.0))
        scaling = context.get("scaling")
        t = context.get("t")

        if u_obs is None or s_obs is None:
            raise ValueError("Both u_obs and s_obs must be provided in the context")

        # Validate inputs
        validation_result = self.validate_inputs(
            u=u_obs,
            s=s_obs,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            scaling=scaling,
            t=t,
        )

        if validation_result.is_error():
            raise ValueError(
                f"Error in dynamics model forward pass: {validation_result.error}"
            )

        # Call implementation
        u_expected, s_expected = self._forward_impl(
            u_obs, s_obs, alpha, beta, gamma, scaling, t
        )

        # Update context with expected counts
        context["u_expected"] = u_expected
        context["s_expected"] = s_expected

        return context

    @abc.abstractmethod
    def _forward_impl(
        self,
        u: BatchTensor,
        s: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        t: Optional[BatchTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Implementation of the forward method.

        This method should be implemented by subclasses to provide the specific
        dynamics model implementation.

        Args:
            u: Observed unspliced RNA counts
            s: Observed spliced RNA counts
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics
            t: Optional time points for the dynamics

        Returns:
            Tuple of (expected unspliced counts, expected spliced counts)
        """
        pass

    @beartype
    def predict_future_states(
        self,
        current_state: Tuple[BatchTensor, BatchTensor],
        time_delta: Union[float, BatchTensor],
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Predict future states based on current state and time delta.

        Args:
            current_state: Tuple of (current unspliced counts, current spliced counts)
            time_delta: Time difference for prediction
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics

        Returns:
            Tuple of (predicted unspliced counts, predicted spliced counts)
        """
        # Validate inputs
        validation_result = self.validate_inputs(
            current_state=current_state,
            time_delta=time_delta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            scaling=scaling,
        )

        if validation_result.is_error():
            raise ValueError(
                f"Error in dynamics model prediction: {validation_result.error}"
            )

        # Call implementation
        return self._predict_future_states_impl(
            current_state, time_delta, alpha, beta, gamma, scaling
        )

    @abc.abstractmethod
    def _predict_future_states_impl(
        self,
        current_state: Tuple[BatchTensor, BatchTensor],
        time_delta: Union[float, BatchTensor],
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Implementation of the predict_future_states method.

        This method should be implemented by subclasses to provide the specific
        future state prediction implementation.

        Args:
            current_state: Tuple of (current unspliced counts, current spliced counts)
            time_delta: Time difference for prediction
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics

        Returns:
            Tuple of (predicted unspliced counts, predicted spliced counts)
        """
        pass

    @beartype
    def steady_state(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the steady-state unspliced and spliced RNA counts.

        Args:
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        # Validate inputs
        validation_result = self.validate_inputs(
            alpha=alpha, beta=beta, gamma=gamma, **kwargs
        )

        if validation_result.is_error():
            raise ValueError(
                f"Error in dynamics model steady state calculation: {validation_result.error}"
            )

        # Call implementation
        return self._steady_state_impl(alpha, beta, gamma, **kwargs)

    @abc.abstractmethod
    def _steady_state_impl(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation of the steady_state method.

        This method should be implemented by subclasses to provide the specific
        steady-state calculation implementation.

        Args:
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        pass


class PyroBufferMixin:
    """
    Mixin class that provides PyroModule's register_buffer functionality.

    This mixin is used to add the register_buffer method to classes without
    causing metaclass conflicts.
    """

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """
        Register a buffer with the module.

        This method mimics PyroModule's register_buffer method by storing
        a tensor as an attribute of the module.

        Args:
            name: The name to register the buffer under.
            tensor: The tensor to register.
        """
        setattr(self, name, tensor)


class BasePriorModel(BaseComponent, PriorModel, PyroBufferMixin, abc.ABC):
    """
    Base class for prior models that define parameter distributions.

    This class implements the PriorModel protocol and provides common
    functionality for prior models. It uses PyroBufferMixin to provide
    the register_buffer method needed by prior model implementations.
    """

    def __init__(self, name: str = "prior_model"):
        """
        Initialize the prior model.

        Args:
            name: A unique name for this component instance.
        """
        super().__init__(name=name)

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, and other parameters

        Returns:
            Updated context dictionary with sampled parameters
        """
        # Extract u_obs and s_obs from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")

        if u_obs is None or s_obs is None:
            raise ValueError("Both u_obs and s_obs must be provided in the context")

        # Create a plate for batched sampling
        n_genes = u_obs.shape[1]
        with pyro.plate(f"{self.name}_plate", n_genes):
            # Sample parameters
            params = self.sample_parameters(n_genes=n_genes)

        # Update the context with the sampled parameters
        context.update(params)

        return context

    @beartype
    def register_priors(self, prefix: str = "") -> None:
        """
        Register prior distributions with Pyro.

        Args:
            prefix: Optional prefix for parameter names
        """
        self._register_priors_impl(prefix)

    @abc.abstractmethod
    def _register_priors_impl(self, prefix: str = "") -> None:
        """
        Implementation of prior registration.

        This method should be implemented by subclasses to provide the specific
        prior registration implementation.

        Args:
            prefix: Optional prefix for parameter names
        """
        pass

    @beartype
    def sample_parameters(
        self, prefix: str = "", n_genes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sample parameters from prior distributions.

        Args:
            prefix: Optional prefix for parameter names
            n_genes: Optional number of genes to sample parameters for

        Returns:
            Dictionary of sampled parameters or a tuple with None and the error
            if sampling fails.
        """
        try:
            return self._sample_parameters_impl(prefix, n_genes)
        except Exception as e:
            # Log the error
            print(f"Error sampling parameters: {e}")
            # Raise a ValueError with a standard message
            raise ValueError(f"Failed to sample parameters") from e

    @abc.abstractmethod
    def _sample_parameters_impl(
        self, prefix: str = "", n_genes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Implementation of parameter sampling.

        This method should be implemented by subclasses to provide the specific
        parameter sampling implementation.

        Args:
            prefix: Optional prefix for parameter names
            n_genes: Optional number of genes to sample parameters for

        Returns:
            Dictionary of sampled parameters
        """
        pass


class BaseLikelihoodModel(BaseComponent, LikelihoodModel, abc.ABC):
    """
    Base class for likelihood models that define observation distributions.

    This class implements the LikelihoodModel protocol and provides common
    functionality for likelihood models.
    """

    def __init__(self, name: str = "likelihood_model"):
        """
        Initialize the likelihood model.

        Args:
            name: A unique name for this component instance.
        """
        super().__init__(name=name)

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define the likelihood distributions for observed data given expected values.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, u_expected, s_expected, and other parameters

        Returns:
            Updated context dictionary with likelihood information
        """
        # Extract required values from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")
        u_expected = context.get("u_expected")
        s_expected = context.get("s_expected")

        if u_obs is None or s_obs is None:
            raise ValueError("Both u_obs and s_obs must be provided in the context")

        if u_expected is None or s_expected is None:
            raise ValueError("Both u_expected and s_expected must be provided in the context")

        # Create a plate for batched sampling
        n_genes = u_obs.shape[1]
        with pyro.plate(f"{self.name}_plate", n_genes):
            # Calculate log probabilities
            u_log_prob = self.log_prob(u_obs, u_expected)
            s_log_prob = self.log_prob(s_obs, s_expected)

        # Update the context with the log probabilities
        context["u_log_prob"] = u_log_prob
        context["s_log_prob"] = s_log_prob

        return context

    @beartype
    def log_prob(
        self,
        observations: torch.Tensor,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate log probability of observations given predictions.

        Args:
            observations: Observed gene expression
            predictions: Predicted gene expression from dynamics model
            scale_factors: Optional scaling factors for observations

        Returns:
            Log probability of observations
        """
        return self._log_prob_impl(observations, predictions, scale_factors)

    @abc.abstractmethod
    def _log_prob_impl(
        self,
        observations: torch.Tensor,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implementation of log probability calculation.

        This method should be implemented by subclasses to provide the specific
        log probability calculation implementation.

        Args:
            observations: Observed gene expression
            predictions: Predicted gene expression from dynamics model
            scale_factors: Optional scaling factors for observations

        Returns:
            Log probability of observations
        """
        pass

    @beartype
    def sample(
        self,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample observations from the likelihood model.

        Args:
            predictions: Predicted gene expression from dynamics model
            scale_factors: Optional scaling factors for observations

        Returns:
            Sampled observations
        """
        return self._sample_impl(predictions, scale_factors)

    @abc.abstractmethod
    def _sample_impl(
        self,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implementation of observation sampling.

        This method should be implemented by subclasses to provide the specific
        sampling implementation.

        Args:
            predictions: Predicted gene expression from dynamics model
            scale_factors: Optional scaling factors for observations

        Returns:
            Sampled observations
        """
        pass


class BaseObservationModel(BaseComponent, ObservationModel, abc.ABC):
    """
    Base class for observation models that transform raw data.

    This class implements the ObservationModel protocol and provides common
    functionality for observation models.
    """

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform observed data for model input.

        Args:
            context: Dictionary containing model context including u_obs and s_obs

        Returns:
            Updated context dictionary with transformed data
        """
        # Extract u_obs and s_obs from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")

        if u_obs is None or s_obs is None:
            raise ValueError("Both u_obs and s_obs must be provided in the context")

        # Create a copy of the context without u_obs and s_obs to avoid duplicate arguments
        context_copy = {k: v for k, v in context.items() if k not in ["u_obs", "s_obs"]}

        # Call the implementation method
        result = self._forward_impl(u_obs, s_obs, **context_copy)

        # Update the context with the result
        context.update(result)

        return context

    @abc.abstractmethod
    def _forward_impl(self, u_obs: torch.Tensor, s_obs: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        """
        Implementation of the forward transformation.

        Args:
            u_obs: Raw observed unspliced RNA counts
            s_obs: Raw observed spliced RNA counts
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with transformed data
        """
        pass

    def __init__(self, name: str = "observation_model"):
        """
        Initialize the observation model.

        Args:
            name: A unique name for this component instance.
        """
        super().__init__(name=name)

    @beartype
    def prepare_data(
        self, adata: AnnData, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data from AnnData object.

        Args:
            adata: AnnData object containing the data
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of prepared data
        """
        try:
            return self._prepare_data_impl(adata, **kwargs)
        except Exception as e:
            # Log the error
            print(f"Error preparing data: {e}")
            # Raise a ValueError with a standard message
            raise ValueError(f"Failed to prepare data") from e

    @abc.abstractmethod
    def _prepare_data_impl(
        self, adata: AnnData, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of data preparation.

        This method should be implemented by subclasses to provide the specific
        data preparation implementation.

        Args:
            adata: AnnData object containing the data
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of prepared data
        """
        pass

    @beartype
    def create_dataloaders(
        self, data: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create data loaders from prepared data.

        Args:
            data: Dictionary of prepared data
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of data loaders
        """
        try:
            return self._create_dataloaders_impl(data, **kwargs)
        except Exception as e:
            # Log the error
            print(f"Error creating dataloaders: {e}")
            # Raise a ValueError with a standard message
            raise ValueError(f"Failed to create dataloaders") from e

    @abc.abstractmethod
    def _create_dataloaders_impl(
        self, data: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Implementation of data loader creation.

        This method should be implemented by subclasses to provide the specific
        data loader creation implementation.

        Args:
            data: Dictionary of prepared data
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of data loaders
        """
        pass

    @beartype
    def preprocess_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of data.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Preprocessed batch data
        """
        try:
            return self._preprocess_batch_impl(batch)
        except Exception as e:
            # Log the error
            print(f"Error preprocessing batch: {e}")
            # Raise a ValueError with a standard message
            raise ValueError(f"Failed to preprocess batch") from e

    @abc.abstractmethod
    def _preprocess_batch_impl(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of batch preprocessing.

        This method should be implemented by subclasses to provide the specific
        batch preprocessing implementation.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Preprocessed batch data
        """
        pass


class BaseInferenceGuide(BaseComponent, InferenceGuide, abc.ABC):
    """
    Base class for inference guides that define approximate posterior distributions.

    This class implements the InferenceGuide protocol and provides common
    functionality for inference guides.
    """

    def __init__(self, name: str = "inference_guide"):
        """
        Initialize the inference guide.

        Args:
            name: A unique name for this component instance.
        """
        super().__init__(name=name)

    @beartype
    def setup_guide(self, model: Union[Callable, Any], **kwargs) -> None:
        """
        Set up the inference guide.

        Args:
            model: Model function to guide
            **kwargs: Additional keyword arguments
        """
        try:
            self._setup_guide_impl(model, **kwargs)
        except Exception as e:
            # Log the error
            print(f"Error setting up guide: {e}")
            # Raise a ValueError with a standard message
            raise ValueError(f"Failed to set up guide") from e

    @abc.abstractmethod
    def _setup_guide_impl(self, model: Union[Callable, Any], **kwargs) -> None:
        """
        Implementation of guide setup.

        This method should be implemented by subclasses to provide the specific
        guide setup implementation.

        Args:
            model: Model function to guide
            **kwargs: Additional keyword arguments
        """
        pass

    @beartype
    def sample_posterior(
        self,
        model: Optional[Callable] = None,
        guide: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the posterior distribution.

        Args:
            model: Optional model function
            guide: Optional guide function
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples
        """
        try:
            return self._sample_posterior_impl(**kwargs)
        except Exception as e:
            # Log the error
            print(f"Error sampling from posterior: {e}")
            # Raise a ValueError with a standard message
            raise ValueError(f"Failed to sample from posterior") from e

    @abc.abstractmethod
    def _sample_posterior_impl(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Implementation of posterior sampling.

        This method should be implemented by subclasses to provide the specific
        posterior sampling implementation.

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples
        """
        pass
