"""
Adapters for converting from monolithic to modular architecture.

This module provides adapters for converting from the legacy monolithic PyroVelocity
architecture to the new modular component-based architecture.
"""

from typing import Any, Dict

from anndata import AnnData
from beartype import beartype

from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.modular.factory import (
    DynamicsModelConfig,
    InferenceGuideConfig,
    LikelihoodModelConfig,
    ObservationModelConfig,
    PriorModelConfig,
    PyroVelocityModelConfig,
    create_model,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


class ConfigurationAdapter:
    """
    Adapter for translating legacy configuration parameters to new modular configurations.

    This class converts configuration parameters from the legacy PyroVelocity model
    to the new modular configuration format, enabling seamless transition between
    the two architectures.
    """

    @staticmethod
    @beartype
    def legacy_to_modular_config(
        legacy_params: Dict[str, Any]
    ) -> PyroVelocityModelConfig:
        """
        Convert legacy PyroVelocity parameters to a modular PyroVelocityModelConfig.

        Args:
            legacy_params: Dictionary of parameters from legacy PyroVelocity model

        Returns:
            PyroVelocityModelConfig for the new modular architecture
        """
        # Extract relevant parameters from legacy config
        model_type = legacy_params.get("model_type", "auto")
        guide_type = legacy_params.get("guide_type", "auto")
        likelihood = legacy_params.get("likelihood", "Poisson")
        shared_time = legacy_params.get("shared_time", True)
        t_scale_on = legacy_params.get("t_scale_on", False)
        cell_specific_kinetics = legacy_params.get(
            "cell_specific_kinetics", None
        )
        kinetics_num = legacy_params.get("kinetics_num", None)

        # Map legacy model_type to dynamics model configuration
        dynamics_params = {
            "shared_time": shared_time,
            "t_scale_on": t_scale_on,
            "cell_specific_kinetics": cell_specific_kinetics,
            "kinetics_num": kinetics_num,
        }

        # Map legacy likelihood to likelihood model configuration
        likelihood_name = (
            "poisson" if likelihood == "Poisson" else "negative_binomial"
        )

        # Map legacy guide_type to inference guide configuration
        guide_name = "auto" if guide_type == "auto" else guide_type.lower()

        # Create the modular configuration
        return PyroVelocityModelConfig(
            dynamics_model=DynamicsModelConfig(
                name="standard" if model_type == "auto" else model_type.lower(),
                params=dynamics_params,
            ),
            prior_model=PriorModelConfig(
                name="lognormal",
                params={},
            ),
            likelihood_model=LikelihoodModelConfig(
                name=likelihood_name,
                params={},
            ),
            observation_model=ObservationModelConfig(
                name="standard",
                params={},
            ),
            inference_guide=InferenceGuideConfig(
                name=guide_name,
                params={},
            ),
            metadata=legacy_params,
        )

    @staticmethod
    @beartype
    def modular_to_legacy_config(
        modular_config: PyroVelocityModelConfig
    ) -> Dict[str, Any]:
        """
        Convert a modular PyroVelocityModelConfig to legacy PyroVelocity parameters.

        Args:
            modular_config: PyroVelocityModelConfig for the new modular architecture

        Returns:
            Dictionary of parameters for the legacy PyroVelocity model
        """
        # Start with any metadata that might contain original legacy params
        legacy_params = (
            modular_config.metadata.copy() if modular_config.metadata else {}
        )

        # Map dynamics model configuration to legacy model_type
        dynamics_name = modular_config.dynamics_model.name
        legacy_params["model_type"] = (
            "auto" if dynamics_name == "standard" else dynamics_name
        )

        # Extract dynamics model parameters
        dynamics_params = modular_config.dynamics_model.params
        legacy_params["shared_time"] = dynamics_params.get("shared_time", True)
        legacy_params["t_scale_on"] = dynamics_params.get("t_scale_on", False)
        legacy_params["cell_specific_kinetics"] = dynamics_params.get(
            "cell_specific_kinetics", None
        )
        legacy_params["kinetics_num"] = dynamics_params.get(
            "kinetics_num", None
        )

        # Map likelihood model configuration to legacy likelihood
        likelihood_name = modular_config.likelihood_model.name
        legacy_params["likelihood"] = (
            "Poisson" if likelihood_name == "poisson" else "NegativeBinomial"
        )

        # Map inference guide configuration to legacy guide_type
        guide_name = modular_config.inference_guide.name
        legacy_params["guide_type"] = (
            "auto" if guide_name == "auto" else guide_name
        )

        return legacy_params


@beartype
def convert_legacy_to_modular(
    legacy_model: PyroVelocity,
) -> PyroVelocityModel:
    """
    Convert a legacy PyroVelocity model to a new PyroVelocityModel.

    Args:
        legacy_model: Legacy PyroVelocity model instance

    Returns:
        PyroVelocityModel instance
    """
    # Extract legacy configuration
    legacy_config = legacy_model.init_params_

    # Convert to modular configuration
    modular_config = ConfigurationAdapter.legacy_to_modular_config(
        legacy_config
    )

    # Create modular model
    modular_model = create_model(modular_config)

    # In a real implementation, we would also transfer trained parameters
    # from the legacy model to the modular model

    return modular_model