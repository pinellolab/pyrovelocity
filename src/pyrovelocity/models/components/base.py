"""
Base component classes for PyroVelocity's modular architecture.

This module provides abstract base classes for each component type that adhere to
the Protocol interfaces defined in `interfaces.py`. These base classes include
common functionality and utilities that would be useful across different
implementations of each component.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import pyro
import torch
from anndata import AnnData
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.interfaces import (
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
)


class BaseDynamicsModel(DynamicsModel, abc.ABC):
    """
    Base class for dynamics models that define gene expression evolution over time.
    
    This class implements the DynamicsModel protocol and provides common
    functionality for dynamics models.
    """
    
    def __init__(self, name: str = "dynamics_model"):
        """
        Initialize the dynamics model.
        
        Args:
            name: A unique name for this component instance.
        """
        self.name = name


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


class BasePriorModel(PriorModel, PyroBufferMixin, abc.ABC):
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
        self.name = name
    
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
    def sample_parameters(self, prefix: str = "") -> Dict[str, Any]:
        """
        Sample parameters from prior distributions.
        
        Args:
            prefix: Optional prefix for parameter names
            
        Returns:
            Dictionary of sampled parameters
        """
        return self._sample_parameters_impl(prefix)
    
    @abc.abstractmethod
    def _sample_parameters_impl(self, prefix: str = "") -> Dict[str, Any]:
        """
        Implementation of parameter sampling.
        
        This method should be implemented by subclasses to provide the specific
        parameter sampling implementation.
        
        Args:
            prefix: Optional prefix for parameter names
            
        Returns:
            Dictionary of sampled parameters
        """
        pass


class BaseLikelihoodModel(LikelihoodModel, abc.ABC):
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
        self.name = name
    
    @jaxtyped
    @beartype
    def log_prob(
        self,
        observations: Float[Array, "batch_size genes"],
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size"]:
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
        observations: Float[Array, "batch_size genes"],
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size"]:
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
    
    @jaxtyped
    @beartype
    def sample(
        self,
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size genes"]:
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
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size genes"]:
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


class BaseObservationModel(ObservationModel, abc.ABC):
    """
    Base class for observation models that transform raw data.
    
    This class implements the ObservationModel protocol and provides common
    functionality for observation models.
    """
    
    def __init__(self, name: str = "observation_model"):
        """
        Initialize the observation model.
        
        Args:
            name: A unique name for this component instance.
        """
        self.name = name
    
    @beartype
    def prepare_data(
        self, adata: AnnData, **kwargs: Any
    ) -> Dict[str, Union[torch.Tensor, jnp.ndarray]]:
        """
        Prepare data from AnnData object.
        
        Args:
            adata: AnnData object containing the data
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of prepared data
        """
        return self._prepare_data_impl(adata, **kwargs)
    
    @abc.abstractmethod
    def _prepare_data_impl(
        self, adata: AnnData, **kwargs: Any
    ) -> Dict[str, Union[torch.Tensor, jnp.ndarray]]:
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
        return self._create_dataloaders_impl(data, **kwargs)
    
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
    ) -> Dict[str, Union[torch.Tensor, jnp.ndarray]]:
        """
        Preprocess a batch of data.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Preprocessed batch data
        """
        return self._preprocess_batch_impl(batch)
    
    @abc.abstractmethod
    def _preprocess_batch_impl(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, jnp.ndarray]]:
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


class BaseInferenceGuide(InferenceGuide, abc.ABC):
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
        self.name = name
    
    @beartype
    def setup_guide(self, model: Callable, **kwargs) -> None:
        """
        Set up the inference guide.
        
        Args:
            model: Model function to guide
            **kwargs: Additional keyword arguments
        """
        self._setup_guide_impl(model, **kwargs)
    
    @abc.abstractmethod
    def _setup_guide_impl(self, model: Callable, **kwargs) -> None:
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
        self, model: Callable, guide: Callable, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the posterior distribution.
        
        Args:
            model: Model function
            guide: Guide function
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of posterior samples
        """
        return self._sample_posterior_impl(model, guide, **kwargs)
    
    @abc.abstractmethod
    def _sample_posterior_impl(
        self, model: Callable, guide: Callable, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of posterior sampling.
        
        This method should be implemented by subclasses to provide the specific
        posterior sampling implementation.
        
        Args:
            model: Model function
            guide: Guide function
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of posterior samples
        """
        pass