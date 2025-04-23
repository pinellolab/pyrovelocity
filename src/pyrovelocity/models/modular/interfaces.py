"""
Core interfaces for PyroVelocity's modular architecture.

This module defines the Protocol interfaces for the different component types
in the PyroVelocity modular architecture. These interfaces establish the contract
that component implementations must follow.
"""

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import pyro
import pyro.distributions as dist
import torch
from beartype.typing import Callable
from jaxtyping import Array, Float, Int

# Type aliases for improved readability
Tensor = torch.Tensor
BatchTensor = Float[Tensor, "batch ..."]
ParamTensor = Float[Tensor, "..."]
CountMatrix = Int[Tensor, "batch cells genes"]
RateMatrix = Float[Tensor, "batch cells genes"]
ModelState = Dict[str, Any]


@runtime_checkable
class DynamicsModel(Protocol):
    """
    Protocol for dynamics models that define the RNA velocity equations.

    Dynamics models define the mathematical relationships between transcription,
    splicing, and degradation rates in the RNA velocity model. They implement
    analytical or numerical solutions to the RNA velocity differential equations.
    """

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
        ...

    def steady_state(
        self,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        **kwargs: Any,
    ) -> Tuple[ParamTensor, ParamTensor]:
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
        ...


@runtime_checkable
class PriorModel(Protocol):
    """
    Protocol for prior models that define the prior distributions for model parameters.

    Prior models define the prior distributions for the model parameters (alpha, beta, gamma)
    and any additional parameters required by the model.
    """

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, and other parameters

        Returns:
            Updated context dictionary with sampled parameters
        """
        ...


@runtime_checkable
class LikelihoodModel(Protocol):
    """
    Protocol for likelihood models that define the observation distributions.

    Likelihood models define how the expected RNA counts (from the dynamics model)
    relate to the observed counts through probability distributions.
    """

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Define the likelihood distributions for observed data given expected values.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, u_expected, s_expected, and other parameters

        Returns:
            Updated context dictionary with likelihood information
        """
        ...


@runtime_checkable
class ObservationModel(Protocol):
    """
    Protocol for observation models that transform raw data for the model.

    Observation models handle data preprocessing, transformation, and normalization
    before feeding it into the dynamics and likelihood models.
    """

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transform observed data for model input.

        Args:
            context: Dictionary containing model context including u_obs and s_obs

        Returns:
            Updated context dictionary with transformed data
        """
        ...


@runtime_checkable
class InferenceGuide(Protocol):
    """
    Protocol for inference guides that define the variational distribution.

    Inference guides define the variational distribution used for approximate
    Bayesian inference in the model.
    """

    def __call__(
        self,
        model: Union[Callable, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        """
        Create a guide function for the given model.

        Args:
            model: The model function to create a guide for
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            A guide function compatible with the model
        """
        ...
