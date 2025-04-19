"""
Factory methods and configuration management for PyroVelocity's modular architecture.

This module provides factory functions for creating PyroVelocity models from
configuration objects, leveraging the registry system and hydra-zen for type-safe
configuration management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from beartype import beartype
from hydra_zen import builds, make_config, make_custom_builds_fn
from omegaconf import DictConfig, OmegaConf

from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
)
from pyrovelocity.models.modular.model import PyroVelocityModel
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
)


# Type-safe configuration dataclasses for each component type
@dataclass
class DynamicsModelConfig:
    """Configuration for dynamics models."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorModelConfig:
    """Configuration for prior models."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LikelihoodModelConfig:
    """Configuration for likelihood models."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationModelConfig:
    """Configuration for observation models."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceGuideConfig:
    """Configuration for inference guides."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PyroVelocityModelConfig:
    """Configuration for the PyroVelocityModel."""

    dynamics_model: DynamicsModelConfig
    prior_model: PriorModelConfig
    likelihood_model: LikelihoodModelConfig
    observation_model: ObservationModelConfig
    inference_guide: InferenceGuideConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


# Create hydra-zen builds for each component type
# These functions create configuration objects with type checking
zen_builds = make_custom_builds_fn(populate_full_signature=True)

DynamicsModelConf = zen_builds(
    DynamicsModelConfig,
    name=str,
    params=dict,
)

PriorModelConf = zen_builds(
    PriorModelConfig,
    name=str,
    params=dict,
)

LikelihoodModelConf = zen_builds(
    LikelihoodModelConfig,
    name=str,
    params=dict,
)

ObservationModelConf = zen_builds(
    ObservationModelConfig,
    name=str,
    params=dict,
)

InferenceGuideConf = zen_builds(
    InferenceGuideConfig,
    name=str,
    params=dict,
)

PyroVelocityModelConf = zen_builds(
    PyroVelocityModelConfig,
    dynamics_model=DynamicsModelConf,
    prior_model=PriorModelConf,
    likelihood_model=LikelihoodModelConf,
    observation_model=ObservationModelConf,
    inference_guide=InferenceGuideConf,
    metadata=dict,
)


@beartype
def create_dynamics_model(
    config: Union[DynamicsModelConfig, Dict, DictConfig]
) -> DynamicsModel:
    """
    Create a dynamics model from a configuration.

    Args:
        config: Configuration for the dynamics model, either as a DynamicsModelConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of the specified dynamics model.

    Raises:
        ValueError: If the specified model is not registered.
    """
    # Convert config to a dictionary if it's a DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = DynamicsModelConfig(**config_dict)
    elif isinstance(config, dict):
        config = DynamicsModelConfig(**config)

    # Create the model using the registry
    return DynamicsModelRegistry.create(config.name, **config.params)


@beartype
def create_prior_model(
    config: Union[PriorModelConfig, Dict, DictConfig]
) -> PriorModel:
    """
    Create a prior model from a configuration.

    Args:
        config: Configuration for the prior model, either as a PriorModelConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of the specified prior model.

    Raises:
        ValueError: If the specified model is not registered.
    """
    # Convert config to a dictionary if it's a DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = PriorModelConfig(**config_dict)
    elif isinstance(config, dict):
        config = PriorModelConfig(**config)

    # Create the model using the registry
    return PriorModelRegistry.create(config.name, **config.params)


@beartype
def create_likelihood_model(
    config: Union[LikelihoodModelConfig, Dict, DictConfig]
) -> LikelihoodModel:
    """
    Create a likelihood model from a configuration.

    Args:
        config: Configuration for the likelihood model, either as a LikelihoodModelConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of the specified likelihood model.

    Raises:
        ValueError: If the specified model is not registered.
    """
    # Convert config to a dictionary if it's a DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = LikelihoodModelConfig(**config_dict)
    elif isinstance(config, dict):
        config = LikelihoodModelConfig(**config)

    # Create the model using the registry
    return LikelihoodModelRegistry.create(config.name, **config.params)


@beartype
def create_observation_model(
    config: Union[ObservationModelConfig, Dict, DictConfig]
) -> ObservationModel:
    """
    Create an observation model from a configuration.

    Args:
        config: Configuration for the observation model, either as an ObservationModelConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of the specified observation model.

    Raises:
        ValueError: If the specified model is not registered.
    """
    # Convert config to a dictionary if it's a DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = ObservationModelConfig(**config_dict)
    elif isinstance(config, dict):
        config = ObservationModelConfig(**config)

    # Create the model using the registry
    return ObservationModelRegistry.create(config.name, **config.params)


@beartype
def create_inference_guide(
    config: Union[InferenceGuideConfig, Dict, DictConfig]
) -> InferenceGuide:
    """
    Create an inference guide from a configuration.

    Args:
        config: Configuration for the inference guide, either as an InferenceGuideConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of the specified inference guide.

    Raises:
        ValueError: If the specified guide is not registered.
    """
    # Convert config to a dictionary if it's a DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = InferenceGuideConfig(**config_dict)
    elif isinstance(config, dict):
        config = InferenceGuideConfig(**config)

    # Create the guide using the registry
    return InferenceGuideRegistry.create(config.name, **config.params)


@beartype
def create_model(
    config: Union[PyroVelocityModelConfig, Dict, DictConfig]
) -> PyroVelocityModel:
    """
    Create a PyroVelocityModel from a configuration.

    This function creates a PyroVelocityModel by instantiating each component
    from the provided configuration and composing them together.

    Args:
        config: Configuration for the PyroVelocityModel, either as a PyroVelocityModelConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of PyroVelocityModel with the specified components.

    Raises:
        ValueError: If any of the specified components are not registered.
    """
    # Convert config to a dictionary if it's a DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = PyroVelocityModelConfig(**config_dict)
    elif isinstance(config, dict):
        config = PyroVelocityModelConfig(**config)

    # Create each component
    dynamics_model = create_dynamics_model(config.dynamics_model)
    prior_model = create_prior_model(config.prior_model)
    likelihood_model = create_likelihood_model(config.likelihood_model)
    observation_model = create_observation_model(config.observation_model)
    inference_guide = create_inference_guide(config.inference_guide)

    # Create and return the model
    return PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        observation_model=observation_model,
        guide_model=inference_guide,
    )


# Predefined configurations for common model setups
def standard_model_config() -> PyroVelocityModelConfig:
    """
    Create a configuration for a standard PyroVelocityModel.

    This function returns a configuration for a PyroVelocityModel with standard
    components: StandardDynamicsModel, LogNormalPriorModel, PoissonLikelihoodModel,
    StandardObservationModel, and AutoGuide.

    Returns:
        A PyroVelocityModelConfig object with standard component configurations.
    """
    return PyroVelocityModelConfig(
        dynamics_model=DynamicsModelConfig(name="standard"),
        prior_model=PriorModelConfig(name="lognormal"),
        likelihood_model=LikelihoodModelConfig(name="poisson"),
        observation_model=ObservationModelConfig(name="standard"),
        inference_guide=InferenceGuideConfig(name="auto"),
    )


def create_standard_model() -> PyroVelocityModel:
    """
    Create a standard PyroVelocityModel.

    This function creates a PyroVelocityModel with standard components:
    StandardDynamicsModel, LogNormalPriorModel, PoissonLikelihoodModel,
    StandardObservationModel, and AutoGuide.

    Returns:
        A PyroVelocityModel instance with standard components.
    """
    return create_model(standard_model_config())


# Export all public symbols
__all__ = [
    # Configuration classes
    "DynamicsModelConfig",
    "PriorModelConfig",
    "LikelihoodModelConfig",
    "ObservationModelConfig",
    "InferenceGuideConfig",
    "PyroVelocityModelConfig",
    # Hydra-zen configuration builders
    "DynamicsModelConf",
    "PriorModelConf",
    "LikelihoodModelConf",
    "ObservationModelConf",
    "InferenceGuideConf",
    "PyroVelocityModelConf",
    # Factory functions
    "create_dynamics_model",
    "create_prior_model",
    "create_likelihood_model",
    "create_observation_model",
    "create_inference_guide",
    "create_model",
    # Predefined configurations
    "standard_model_config",
    "create_standard_model",
]
