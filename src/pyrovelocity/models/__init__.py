"""PyroVelocity models package.

This package contains the modular components of the PyroVelocity model architecture,
including factory methods for model creation and configuration management, adapters for
backward compatibility with the legacy API, and Bayesian model comparison tools.
"""

# Import from modular components
from pyrovelocity.models.modular.components.base import (
    BaseDynamicsModel,
    BaseLikelihoodModel,
    BaseObservationModel,
    BasePriorModel,
    BaseInferenceGuide,
)
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
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
    InferenceGuide,
)
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
    InferenceGuideRegistry,
)

# Import the factory module for model creation and configuration
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

# Import the legacy PyroVelocity model for backward compatibility
from pyrovelocity.models._velocity import PyroVelocity

# Import the new PyroVelocityModel and ModelState
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel

# Import the adapter classes and functions for backward compatibility
from pyrovelocity.models.adapters import (
    ConfigurationAdapter,
    LegacyModelAdapter,
    ModularModelAdapter,
    convert_legacy_to_modular,
    convert_modular_to_legacy,
)

# Import the model comparison classes and functions
from pyrovelocity.models.modular.comparison import (
    BayesianModelComparison,
    ComparisonResult,
    create_comparison_table,
    select_best_model,
)

# Import the model selection classes and functions
from pyrovelocity.models.modular.selection import (
    ModelSelection,
    ModelEnsemble,
    CrossValidator,
    SelectionCriterion,
    SelectionResult,
)

# Import experimental implementations
from pyrovelocity.models.experimental import (
    deterministic_transcription_splicing_probabilistic_model,
    generate_test_data_for_deterministic_model_inference,
    generate_prior_inference_data,
    generate_posterior_inference_data,
    plot_sample_phase_portraits,
    plot_sample_trajectories,
    plot_sample_trajectories_with_percentiles,
    save_inference_plots,
    solve_transcription_splicing_model,
    solve_transcription_splicing_model_analytical,
    lognormal_tail_probability,
    solve_for_lognormal_sigma_given_threshold_and_tail_mass,
    solve_for_lognormal_mu_given_threshold_and_tail_mass,
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
    # Experimental implementations
    "deterministic_transcription_splicing_probabilistic_model",
    "generate_test_data_for_deterministic_model_inference",
    "generate_prior_inference_data",
    "generate_posterior_inference_data",
    "plot_sample_phase_portraits",
    "plot_sample_trajectories",
    "plot_sample_trajectories_with_percentiles",
    "save_inference_plots",
    "solve_transcription_splicing_model",
    "solve_transcription_splicing_model_analytical",
    "lognormal_tail_probability",
    "solve_for_lognormal_sigma_given_threshold_and_tail_mass",
    "solve_for_lognormal_mu_given_threshold_and_tail_mass",
]
