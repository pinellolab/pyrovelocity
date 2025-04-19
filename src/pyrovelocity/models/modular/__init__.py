"""
PyroVelocity modular model architecture.

This package contains the modular components of the PyroVelocity model architecture,
including component interfaces, implementations, factory methods, and registries.
"""

# Import component interfaces
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
    InferenceGuide,
)

# Import component base classes
from pyrovelocity.models.modular.components.base import (
    BaseDynamicsModel,
    BaseLikelihoodModel,
    BaseObservationModel,
    BasePriorModel,
    BaseInferenceGuide,
)

# Import component implementations
from pyrovelocity.models.modular.components.dynamics import (
    StandardDynamicsModel,
    NonlinearDynamicsModel,
)
from pyrovelocity.models.modular.components.priors import (
    LogNormalPriorModel,
    InformativePriorModel,
)
from pyrovelocity.models.modular.components.likelihoods import (
    PoissonLikelihoodModel,
    NegativeBinomialLikelihoodModel,
)
from pyrovelocity.models.modular.components.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    NormalGuide,
    DeltaGuide,
)

# Import component registries
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
    InferenceGuideRegistry,
)

# Import factory methods
from pyrovelocity.models.modular.factory import (
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

# Import model classes
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel

# Import model comparison tools
from pyrovelocity.models.modular.comparison import (
    BayesianModelComparison,
    ComparisonResult,
    create_comparison_table,
    select_best_model,
)

# Import model selection tools
from pyrovelocity.models.modular.selection import (
    ModelSelection,
    ModelEnsemble,
    CrossValidator,
    SelectionCriterion,
    SelectionResult,
)

__all__ = [
    # Interfaces
    "DynamicsModel",
    "LikelihoodModel",
    "ObservationModel",
    "PriorModel",
    "InferenceGuide",
    # Base classes
    "BaseDynamicsModel",
    "BaseLikelihoodModel",
    "BaseObservationModel",
    "BasePriorModel",
    "BaseInferenceGuide",
    # Component implementations
    "StandardDynamicsModel",
    "NonlinearDynamicsModel",
    "LogNormalPriorModel",
    "InformativePriorModel",
    "PoissonLikelihoodModel",
    "NegativeBinomialLikelihoodModel",
    "StandardObservationModel",
    "AutoGuideFactory",
    "NormalGuide",
    "DeltaGuide",
    # Registries
    "DynamicsModelRegistry",
    "LikelihoodModelRegistry",
    "ObservationModelRegistry",
    "PriorModelRegistry",
    "InferenceGuideRegistry",
    # Factory methods
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
    # Model classes
    "ModelState",
    "PyroVelocityModel",
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
