"""PyroVelocity models package.

This package contains the modular components of the PyroVelocity model architecture,
including factory methods for model creation and configuration management, adapters for
backward compatibility with the legacy API, and Bayesian model comparison tools.
"""

from pyrovelocity.models.components.base import (
    BaseDynamicsModel,
    BaseLikelihoodModel,
    BaseObservationModel,
    BasePriorModel,
    BaseInferenceGuide,
)
from pyrovelocity.models.components.dynamics import (
    StandardDynamicsModel,
    NonlinearDynamicsModel,
)
from pyrovelocity.models.components.priors import (
    LogNormalPriorModel,
    InformativePriorModel,
)
from pyrovelocity.models.components.likelihoods import (
    PoissonLikelihoodModel,
    NegativeBinomialLikelihoodModel,
)
from pyrovelocity.models.components.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.components.guides import (
    AutoGuideFactory,
    NormalGuide,
    DeltaGuide,
)
from pyrovelocity.models.interfaces import (
    DynamicsModel,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
    InferenceGuide,
)
from pyrovelocity.models.registry import (
    DynamicsModelRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
    InferenceGuideRegistry,
)

# Import the factory module for model creation and configuration
from pyrovelocity.models.factory import (
    DynamicsModelConfig,
    PriorModelConfig,
    LikelihoodModelConfig,
    ObservationModelConfig,
    InferenceGuideConfig,
    PyroVelocityModelConfig,
    create_dynamics_model,
    create_prior_model,
    create_likelihood_model,
    create_observation_model,
    create_inference_guide,
    create_model,
    standard_model_config,
    create_standard_model,
)

# Import the legacy PyroVelocity model for backward compatibility
from pyrovelocity.models._velocity import PyroVelocity

# Import the new PyroVelocityModel and ModelState
from pyrovelocity.models.model import ModelState, PyroVelocityModel

# Import the adapter classes and functions for backward compatibility
from pyrovelocity.models.adapters import (
    ConfigurationAdapter,
    LegacyModelAdapter,
    ModularModelAdapter,
    convert_legacy_to_modular,
    convert_modular_to_legacy,
)

# Import the model comparison classes and functions
from pyrovelocity.models.comparison import (
    BayesianModelComparison,
    ComparisonResult,
    create_comparison_table,
    select_best_model,
)

# Import the model selection classes and functions
from pyrovelocity.models.selection import (
    ModelSelection,
    ModelEnsemble,
    CrossValidator,
    SelectionCriterion,
    SelectionResult,
)

__all__ = [
    # Base classes
    "BaseDynamicsModel",
    "BaseLikelihoodModel",
    "BaseObservationModel",
    "BasePriorModel",
    "BaseInferenceGuide",
    
    # Interfaces
    "DynamicsModel",
    "LikelihoodModel",
    "ObservationModel",
    "PriorModel",
    "InferenceGuide",
    
    # Registries
    "DynamicsModelRegistry",
    "LikelihoodModelRegistry",
    "ObservationModelRegistry",
    "PriorModelRegistry",
    "InferenceGuideRegistry",
    
    # Dynamics models
    "StandardDynamicsModel",
    "NonlinearDynamicsModel",
    
    # Prior models
    "LogNormalPriorModel",
    "InformativePriorModel",
    
    # Likelihood models
    "PoissonLikelihoodModel",
    "NegativeBinomialLikelihoodModel",
    
    # Observation models
    "StandardObservationModel",
    
    # Inference guides
    "AutoGuideFactory",
    "NormalGuide",
    "DeltaGuide",
    
    # Legacy model
    "PyroVelocity",
    
    # New model
    "ModelState",
    "PyroVelocityModel",
    
    # Adapter classes and functions
    "ConfigurationAdapter",
    "LegacyModelAdapter",
    "ModularModelAdapter",
    "convert_legacy_to_modular",
    "convert_modular_to_legacy",
    
    # Factory module
    "DynamicsModelConfig",
    "PriorModelConfig",
    "LikelihoodModelConfig",
    "ObservationModelConfig",
    "InferenceGuideConfig",
    "PyroVelocityModelConfig",
    "create_dynamics_model",
    "create_prior_model",
    "create_likelihood_model",
    "create_observation_model",
    "create_inference_guide",
    "create_model",
    "standard_model_config",
    "create_standard_model",
    
    # Model comparison
    "BayesianModelComparison",
    "ComparisonResult",
    "create_comparison_table",
    "select_best_model",
    
    # Model selection
    "ModelSelection",
    "ModelEnsemble",
    "CrossValidator",
    "SelectionCriterion",
    "SelectionResult",
]
