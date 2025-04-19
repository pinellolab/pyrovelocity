"""
PyroVelocity model implementation that composes component models.
    
This module provides the core model class for PyroVelocity's modular architecture,
implementing a composable approach where the full model is built from specialized
component models (dynamics, priors, likelihoods, observations, guides).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Protocol

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, Int

from pyrovelocity.models.modular.components.base import BaseDynamicsModel
from pyrovelocity.models.modular.components.guides import AutoGuideFactory
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel as DynamicsModelProtocol,
    InferenceGuide as GuideModelProtocol,
    LikelihoodModel as LikelihoodModelProtocol,
    ObservationModel as ObservationModelProtocol,
    PriorModel as PriorModelProtocol,
)
from pyrovelocity.models.modular.components.likelihoods import (
    PoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.modular.components.priors import LogNormalPriorModel


@dataclass(frozen=True)
class ModelState:
    """
    Immutable state container for the PyroVelocityModel.

    This dataclass holds the state of all component models and ensures immutability
    through the frozen=True parameter.

    Attributes:
        dynamics_state: State of the dynamics model component
        prior_state: State of the prior model component
        likelihood_state: State of the likelihood model component
        observation_state: State of the observation model component
        guide_state: State of the inference guide component
        metadata: Optional dictionary for additional metadata
    """

    dynamics_state: Dict[str, Any]
    prior_state: Dict[str, Any]
    likelihood_state: Dict[str, Any]
    observation_state: Dict[str, Any]
    guide_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate state after initialization."""
        # Since the class is frozen, we can't modify attributes directly
        # This method is for validation only
        pass


class PyroVelocityModel:
    """
    Composable PyroVelocity model that integrates specialized component models.

    This class implements the core model for PyroVelocity's modular architecture,
    composing specialized component models (dynamics, priors, likelihoods,
    observations, guides) into a cohesive probabilistic model. It uses functional
    composition for the forward method, enabling railway-oriented programming
    patterns.

    The model follows a clear separation of concerns:
    - Dynamics model: Defines the velocity vector field
    - Prior model: Specifies priors for model parameters
    - Likelihood model: Defines the observation likelihood
    - Observation model: Handles data preprocessing and transformation
    - Guide model: Implements the inference guide for posterior approximation

    Attributes:
        dynamics_model: Component handling velocity vector field modeling
        prior_model: Component handling prior distributions
        likelihood_model: Component handling likelihood distributions
        observation_model: Component handling data observations
        guide_model: Component handling inference guide
        state: Immutable state container for all model components
    """

    @beartype
    def __init__(
        self,
        dynamics_model: DynamicsModelProtocol,
        prior_model: PriorModelProtocol,
        likelihood_model: LikelihoodModelProtocol,
        observation_model: ObservationModelProtocol,
        guide_model: GuideModelProtocol,
        state: Optional[ModelState] = None,
    ):
        """
        Initialize the PyroVelocityModel with component models.

        Args:
            dynamics_model: Model component for velocity vector field
            prior_model: Model component for prior distributions
            likelihood_model: Model component for likelihood distributions
            observation_model: Model component for data observations
            guide_model: Model component for inference guide
            state: Optional pre-initialized model state
        """
        # Store the component models
        self.dynamics_model = dynamics_model
        self.prior_model = prior_model
        self.likelihood_model = likelihood_model
        self.observation_model = observation_model
        self.guide_model = guide_model

        # Initialize state if not provided
        if state is None:
            self.state = ModelState(
                dynamics_state=getattr(self.dynamics_model, "state", {}),
                prior_state=getattr(self.prior_model, "state", {}),
                likelihood_state=getattr(self.likelihood_model, "state", {}),
                observation_state=getattr(self.observation_model, "state", {}),
                guide_state=getattr(self.guide_model, "state", {}),
            )
        else:
            self.state = state

    @beartype
    def forward(
        self,
        x: Float[Array, "batch_size n_features"],
        time_points: Float[Array, "n_times"],
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model using functional composition.

        This method implements the forward pass through all model components
        using functional composition, following a railway-oriented programming
        approach where each component processes the data and passes it to the
        next component.

        Args:
            x: Input data tensor of shape [batch_size, n_features]
            time_points: Time points for the dynamics model
            cell_state: Optional dictionary with cell state information
            **kwargs: Additional keyword arguments passed to component models

        Returns:
            Dictionary containing model outputs and intermediate results
        """
        # Initialize the context dictionary to pass between components
        context = {
            "x": x,
            "time_points": time_points,
            "cell_state": cell_state or {},
            **kwargs,
        }

        # Process data through the observation model
        observation_context = self.observation_model.forward(context)

        # Process through the dynamics model
        dynamics_context = self.dynamics_model.forward(observation_context)

        # Apply prior distributions
        prior_context = self.prior_model.forward(dynamics_context)

        # Apply likelihood model
        likelihood_context = self.likelihood_model.forward(prior_context)

        # Return the final context with all model outputs
        return likelihood_context

    @beartype
    def guide(
        self,
        x: Float[Array, "batch_size n_features"],
        time_points: Float[Array, "n_times"],
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inference guide for the model.

        This method delegates to the guide model component to implement
        the inference guide for posterior approximation.

        Args:
            x: Input data tensor of shape [batch_size, n_features]
            time_points: Time points for the dynamics model
            cell_state: Optional dictionary with cell state information
            **kwargs: Additional keyword arguments passed to the guide model

        Returns:
            Dictionary containing guide outputs
        """
        # Initialize the context dictionary to pass to the guide
        context = {
            "x": x,
            "time_points": time_points,
            "cell_state": cell_state or {},
            **kwargs,
        }

        # Process through the observation model first to prepare data
        observation_context = self.observation_model.forward(context)

        # Delegate to the guide model
        return self.guide_model.forward(observation_context)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "PyroVelocityModel"

    def get_state(self) -> ModelState:
        """Return the current model state."""
        return self.state

    def with_state(self, state: ModelState) -> "PyroVelocityModel":
        """
        Create a new model instance with the given state.

        This method implements the immutable state pattern, returning a new
        model instance with the updated state rather than modifying the
        current instance.

        Args:
            state: New model state

        Returns:
            New PyroVelocityModel instance with the given state
        """
        return PyroVelocityModel(
            dynamics_model=self.dynamics_model,
            prior_model=self.prior_model,
            likelihood_model=self.likelihood_model,
            observation_model=self.observation_model,
            guide_model=self.guide_model,
            state=state,
        )
