"""
Factory methods and configuration management for PyroVelocity's modular architecture.

This module provides factory functions for creating PyroVelocity models from
configuration objects, leveraging the registry system and hydra-zen for type-safe
configuration management.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Union, cast

from beartype import beartype
from hydra_zen import make_custom_builds_fn
from omegaconf import DictConfig, OmegaConf

from pyrovelocity.models.modular.config import (
    ComponentConfig,
    ComponentType,
    ModelConfig,
)
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
    register_standard_components,
)


class ComponentFactory:
    """
    Factory for creating components in the PyroVelocity modular architecture.

    This class provides methods for creating components from configurations,
    replacing the multiple component-specific factory functions in the original
    implementation.
    """

    @staticmethod
    @beartype
    def create_component(
        config: ComponentConfig, component_type: ComponentType
    ) -> Union[DynamicsModel, PriorModel, LikelihoodModel, ObservationModel, InferenceGuide]:
        """
        Create a component from a configuration.

        Args:
            config: Configuration for the component
            component_type: Type of component to create

        Returns:
            An instance of the specified component

        Raises:
            ValueError: If the specified component is not registered or the component type is invalid
        """
        # Select the appropriate registry based on the component type
        if component_type == ComponentType.DYNAMICS_MODEL:
            return DynamicsModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.PRIOR_MODEL:
            return PriorModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.LIKELIHOOD_MODEL:
            return LikelihoodModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.OBSERVATION_MODEL:
            return ObservationModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.INFERENCE_GUIDE:
            return InferenceGuideRegistry.create(config.name, **config.params)
        else:
            raise ValueError(f"Invalid component type: {component_type}")

    @staticmethod
    @beartype
    def create_component_from_dict(
        config_dict: Dict[str, Any], component_type: ComponentType
    ) -> Union[DynamicsModel, PriorModel, LikelihoodModel, ObservationModel, InferenceGuide]:
        """
        Create a component from a dictionary configuration.

        Args:
            config_dict: Dictionary containing the configuration
            component_type: Type of component to create

        Returns:
            An instance of the specified component

        Raises:
            ValueError: If the specified component is not registered or the component type is invalid
        """
        config = ComponentConfig.from_dict(config_dict)
        return ComponentFactory.create_component(config, component_type)

    @staticmethod
    @beartype
    def create_dynamics_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> DynamicsModel:
        """
        Create a dynamics model from a configuration.

        Args:
            config: Configuration for the dynamics model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified dynamics model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            DynamicsModel,
            ComponentFactory.create_component(config, ComponentType.DYNAMICS_MODEL),
        )

    @staticmethod
    @beartype
    def create_prior_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> PriorModel:
        """
        Create a prior model from a configuration.

        Args:
            config: Configuration for the prior model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified prior model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            PriorModel,
            ComponentFactory.create_component(config, ComponentType.PRIOR_MODEL),
        )

    @staticmethod
    @beartype
    def create_likelihood_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> LikelihoodModel:
        """
        Create a likelihood model from a configuration.

        Args:
            config: Configuration for the likelihood model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified likelihood model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            LikelihoodModel,
            ComponentFactory.create_component(config, ComponentType.LIKELIHOOD_MODEL),
        )

    @staticmethod
    @beartype
    def create_observation_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> ObservationModel:
        """
        Create an observation model from a configuration.

        Args:
            config: Configuration for the observation model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified observation model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            ObservationModel,
            ComponentFactory.create_component(config, ComponentType.OBSERVATION_MODEL),
        )

    @staticmethod
    @beartype
    def create_inference_guide(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> InferenceGuide:
        """
        Create an inference guide from a configuration.

        Args:
            config: Configuration for the inference guide, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified inference guide.

        Raises:
            ValueError: If the specified guide is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            InferenceGuide,
            ComponentFactory.create_component(config, ComponentType.INFERENCE_GUIDE),
        )


# For backward compatibility, we keep the old configuration classes
# These will be deprecated in a future release

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
    inference_guide=InferenceGuideConf,
    metadata=dict,
)


@beartype
def create_model_from_config(
    config: Union[ModelConfig, Dict[str, Any], DictConfig]
) -> PyroVelocityModel:
    """
    Create a PyroVelocityModel from a configuration.

    This function creates a PyroVelocityModel by instantiating each component
    from the provided configuration and composing them together.

    Args:
        config: Configuration for the PyroVelocityModel, either as a ModelConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of PyroVelocityModel with the specified components.

    Raises:
        ValueError: If any of the specified components are not registered.
    """
    # Convert config to a ModelConfig if it's a dictionary or DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = ModelConfig.from_dict(config_dict)
    elif isinstance(config, dict):
        config = ModelConfig.from_dict(config)

    # Create each component using the ComponentFactory
    dynamics_model = ComponentFactory.create_dynamics_model(config.dynamics_model)
    prior_model = ComponentFactory.create_prior_model(config.prior_model)
    likelihood_model = ComponentFactory.create_likelihood_model(config.likelihood_model)
    inference_guide = ComponentFactory.create_inference_guide(config.inference_guide)

    # Create and return the model (observation model functionality is now in likelihood model)
    return PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        guide_model=inference_guide,
    )


# Backward compatibility functions
# These functions use the new unified configuration system but maintain
# the same interface as the original functions for backward compatibility

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

    # Create the model using the ComponentFactory
    component_config = ComponentConfig(name=config.name, params=config.params)
    return ComponentFactory.create_dynamics_model(component_config)


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

    # Create the model using the ComponentFactory
    component_config = ComponentConfig(name=config.name, params=config.params)
    return ComponentFactory.create_prior_model(component_config)


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

    # Create the model using the ComponentFactory
    component_config = ComponentConfig(name=config.name, params=config.params)
    return ComponentFactory.create_likelihood_model(component_config)


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

    # Create the model using the ComponentFactory
    component_config = ComponentConfig(name=config.name, params=config.params)
    return ComponentFactory.create_observation_model(component_config)


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

    # Create the guide using the ComponentFactory
    component_config = ComponentConfig(name=config.name, params=config.params)
    return ComponentFactory.create_inference_guide(component_config)


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
    # Note: observation_model is ignored as its functionality is now in likelihood_model
    # observation_model = create_observation_model(config.observation_model)
    inference_guide = create_inference_guide(config.inference_guide)

    # Create and return the model (observation model functionality is now in likelihood model)
    return PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        guide_model=inference_guide,
    )


# Predefined configurations for common model setups
def standard_model_config() -> PyroVelocityModelConfig:
    """
    Create a configuration for a standard PyroVelocityModel.

    This function returns a configuration for a PyroVelocityModel with standard
    components: StandardDynamicsModel, LogNormalPriorModel, PoissonLikelihoodModel
    (includes data preprocessing), and AutoGuide.

    Returns:
        A PyroVelocityModelConfig object with standard component configurations.
    """
    # For backward compatibility, we return the old config type
    # In the future, we should encourage users to use ModelConfig.standard() directly
    return PyroVelocityModelConfig(
        dynamics_model=DynamicsModelConfig(name="standard"),
        prior_model=PriorModelConfig(name="lognormal"),
        likelihood_model=LikelihoodModelConfig(name="poisson"),
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
    # Use the new ModelConfig.standard() for creating the model
    return create_model_from_config(ModelConfig.standard())


# Legacy model replication functions
def create_legacy_model1() -> PyroVelocityModel:
    """
    Create a PyroVelocityModel that replicates the legacy model with add_offset=False.

    This function creates a PyroVelocityModel with components specifically configured
    to match the behavior of the legacy PyroVelocity implementation with add_offset=False.
    It uses the LegacyDynamicsModel to create a model that exactly matches the
    behavior of the legacy implementation.

    Returns:
        A PyroVelocityModel instance that replicates the legacy model.
    """
    # Create a configuration with the legacy dynamics model
    config = ModelConfig(
        dynamics_model=ComponentConfig(
            name="legacy",
            params={
                "shared_time": True,
                "t_scale_on": False,
                "correct_library_size": True,
            },
        ),
        prior_model=ComponentConfig(
            name="lognormal",
            params={},
        ),
        likelihood_model=ComponentConfig(
            name="legacy",
            params={},
        ),
        inference_guide=ComponentConfig(
            name="auto",
            params={
                "init_scale": 0.1,
            },
        ),
    )

    # Create and return the model
    return create_model_from_config(config)


def create_legacy_model2() -> PyroVelocityModel:
    """
    Create a PyroVelocityModel that replicates the legacy model with add_offset=True.

    This function creates a PyroVelocityModel with components specifically configured
    to match the behavior of the legacy PyroVelocity implementation with add_offset=True.
    It uses the LegacyDynamicsModel to create a model that exactly matches the
    behavior of the legacy implementation.

    Returns:
        A PyroVelocityModel instance that replicates the legacy model.
    """
    # Create a configuration with the legacy dynamics model
    config = ModelConfig(
        dynamics_model=ComponentConfig(
            name="legacy",
            params={
                "shared_time": True,
                "t_scale_on": False,
                "correct_library_size": True,
                "add_offset": True,
            },
        ),
        prior_model=ComponentConfig(
            name="lognormal",
            params={},
        ),
        likelihood_model=ComponentConfig(
            name="legacy",
            params={},
        ),
        inference_guide=ComponentConfig(
            name="legacy_auto",
            params={
                "init_scale": 0.1,
                "add_offset": True,
            },
        ),
    )

    # Create and return the model
    return create_model_from_config(config)


def create_piecewise_activation_model() -> PyroVelocityModel:
    """
    Create a PyroVelocityModel with piecewise activation components.

    This function creates a PyroVelocityModel specifically configured for
    piecewise activation parameter recovery validation. It uses:
    - PiecewiseActivationDynamicsModel for dimensionless analytical dynamics
    - PiecewiseActivationPriorModel for hierarchical priors
    - PoissonLikelihoodModel for count data likelihood (includes data preprocessing)
    - AutoGuideFactory for variational inference

    Returns:
        A PyroVelocityModel instance configured for piecewise activation validation.

    Examples:
        >>> model = create_piecewise_activation_model()
        >>> # Use model for parameter recovery validation
        >>> # model.train(adata, max_epochs=100)
        >>> # posterior_samples = model.generate_posterior_samples(adata)
    """
    # Create configuration for piecewise activation model
    config = ModelConfig(
        dynamics_model=ComponentConfig(
            name="piecewise_activation",
            params={},
        ),
        prior_model=ComponentConfig(
            name="piecewise_activation",
            params={},
        ),
        likelihood_model=ComponentConfig(
            name="piecewise_activation_poisson",
            params={},
        ),
        inference_guide=ComponentConfig(
            name="auto",
            params={
                "guide_type": "AutoLowRankMultivariateNormal",
                "init_scale": 0.1,
            },
        ),
    )

    # Create and return the model
    return create_model_from_config(config)


# Export all public symbols
__all__ = [
    # Factory class
    "ComponentFactory",
    # New factory functions
    "create_model_from_config",
    # Backward compatibility configuration classes
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
    # Backward compatibility functions
    "create_dynamics_model",
    "create_prior_model",
    "create_likelihood_model",
    "create_observation_model",
    "create_inference_guide",
    "create_model",
    # Legacy model replication functions
    "create_legacy_model1",
    "create_legacy_model2",
    # Predefined configurations
    "standard_model_config",
    "create_standard_model",
    # Piecewise activation model
    "create_piecewise_activation_model",
]

# Register standard components when the module is imported
register_standard_components()
