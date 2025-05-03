"""
Unified configuration system for PyroVelocity's modular architecture.

This module provides a unified configuration system for PyroVelocity's modular
architecture, including a generic ComponentConfig class and a ModelConfig class
that composes ComponentConfig instances.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Type, Union, cast

from beartype import beartype
from hydra_zen import builds, make_config, make_custom_builds_fn
from omegaconf import DictConfig, OmegaConf


class ComponentType(Enum):
    """Enumeration of component types in the PyroVelocity modular architecture."""

    DYNAMICS_MODEL = auto()
    PRIOR_MODEL = auto()
    LIKELIHOOD_MODEL = auto()
    OBSERVATION_MODEL = auto()
    INFERENCE_GUIDE = auto()


@dataclass
class ComponentConfig:
    """
    Generic configuration for components in the PyroVelocity modular architecture.

    This class provides a unified configuration interface for all component types,
    replacing the multiple component-specific configuration classes in the original
    implementation.

    Attributes:
        name: The name of the component to create (must be registered in the appropriate registry)
        params: Parameters to pass to the component constructor
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ComponentConfig":
        """
        Create a ComponentConfig from a dictionary.

        Args:
            config_dict: Dictionary containing the configuration

        Returns:
            A ComponentConfig instance
        """
        return cls(
            name=config_dict["name"],
            params=config_dict.get("params", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ComponentConfig to a dictionary.

        Returns:
            A dictionary representation of the ComponentConfig
        """
        return {
            "name": self.name,
            "params": self.params,
        }


@dataclass
class ModelConfig:
    """
    Configuration for the PyroVelocityModel.

    This class composes ComponentConfig instances for each component type,
    providing a unified configuration interface for the entire model.

    Attributes:
        dynamics_model: Configuration for the dynamics model
        prior_model: Configuration for the prior model
        likelihood_model: Configuration for the likelihood model
        observation_model: Configuration for the observation model
        inference_guide: Configuration for the inference guide
        metadata: Additional metadata for the model
    """

    dynamics_model: ComponentConfig
    prior_model: ComponentConfig
    likelihood_model: ComponentConfig
    observation_model: ComponentConfig
    inference_guide: ComponentConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Create a ModelConfig from a dictionary.

        Args:
            config_dict: Dictionary containing the configuration

        Returns:
            A ModelConfig instance
        """
        return cls(
            dynamics_model=ComponentConfig.from_dict(config_dict["dynamics_model"]),
            prior_model=ComponentConfig.from_dict(config_dict["prior_model"]),
            likelihood_model=ComponentConfig.from_dict(config_dict["likelihood_model"]),
            observation_model=ComponentConfig.from_dict(config_dict["observation_model"]),
            inference_guide=ComponentConfig.from_dict(config_dict["inference_guide"]),
            metadata=config_dict.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ModelConfig to a dictionary.

        Returns:
            A dictionary representation of the ModelConfig
        """
        return {
            "dynamics_model": self.dynamics_model.to_dict(),
            "prior_model": self.prior_model.to_dict(),
            "likelihood_model": self.likelihood_model.to_dict(),
            "observation_model": self.observation_model.to_dict(),
            "inference_guide": self.inference_guide.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def standard(cls, use_protocol_first: bool = False) -> "ModelConfig":
        """
        Create a standard model configuration.

        This method returns a configuration for a PyroVelocityModel with standard
        components: StandardDynamicsModel, LogNormalPriorModel, PoissonLikelihoodModel,
        StandardObservationModel, and AutoGuide.

        Args:
            use_protocol_first: If True, use Protocol-First component implementations
                               (with "_direct" suffix in component names)

        Returns:
            A ModelConfig instance with standard component configurations
        """
        # Add "_direct" suffix to component names if using Protocol-First implementations
        suffix = "_direct" if use_protocol_first else ""

        return cls(
            dynamics_model=ComponentConfig(name=f"standard{suffix}"),
            prior_model=ComponentConfig(name=f"lognormal{suffix}"),
            likelihood_model=ComponentConfig(name=f"poisson{suffix}"),
            observation_model=ComponentConfig(name=f"standard{suffix}"),
            inference_guide=ComponentConfig(name=f"auto{suffix}"),
        )


# Create hydra-zen builds for the configuration classes
zen_builds = make_custom_builds_fn(populate_full_signature=True)

ComponentConfigConf = zen_builds(
    ComponentConfig,
    name=str,
    params=dict,
)

ModelConfigConf = zen_builds(
    ModelConfig,
    dynamics_model=ComponentConfigConf,
    prior_model=ComponentConfigConf,
    likelihood_model=ComponentConfigConf,
    observation_model=ComponentConfigConf,
    inference_guide=ComponentConfigConf,
    metadata=dict,
)


# Export all public symbols
__all__ = [
    # Enums
    "ComponentType",
    # Configuration classes
    "ComponentConfig",
    "ModelConfig",
    # Hydra-zen configuration builders
    "ComponentConfigConf",
    "ModelConfigConf",
]
