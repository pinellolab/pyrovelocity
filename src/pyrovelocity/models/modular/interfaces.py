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
        u: BatchTensor,
        s: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        t: Optional[BatchTensor] = None,
        **kwargs: Any,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Compute the expected unspliced and spliced RNA counts based on the dynamics model.

        Args:
            u: Observed unspliced RNA counts
            s: Observed spliced RNA counts
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics
            t: Optional time points for the dynamics
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (expected unspliced counts, expected spliced counts)
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
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        plate: pyro.plate,
        **kwargs: Any,
    ) -> ModelState:
        """
        Sample model parameters from prior distributions.

        Args:
            u_obs: Observed unspliced RNA counts
            s_obs: Observed spliced RNA counts
            plate: Pyro plate for batched sampling
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing sampled parameters
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
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        u_logits: BatchTensor,
        s_logits: BatchTensor,
        plate: pyro.plate,
        **kwargs: Any,
    ) -> None:
        """
        Define the likelihood distributions for observed data given expected values.

        Args:
            u_obs: Observed unspliced RNA counts
            s_obs: Observed spliced RNA counts
            u_logits: Expected unspliced RNA counts from dynamics model
            s_logits: Expected spliced RNA counts from dynamics model
            plate: Pyro plate for batched sampling
            **kwargs: Additional model-specific parameters
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
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        **kwargs: Any,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Transform observed data for model input.

        Args:
            u_obs: Raw observed unspliced RNA counts
            s_obs: Raw observed spliced RNA counts
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (transformed unspliced counts, transformed spliced counts)
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
        model: Callable,
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
