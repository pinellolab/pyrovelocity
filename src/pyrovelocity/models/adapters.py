"""
Adapter layer for PyroVelocity's modular architecture.

This module provides adapters for backward compatibility with the existing PyroVelocity API,
allowing seamless transition from the legacy monolithic architecture to the new modular
component-based architecture. The adapters implement the Adapter pattern to translate
between old-style function calls and new component-based calls.

The module includes:
1. LegacyModelAdapter - Adapts the new PyroVelocityModel to the legacy PyroVelocity API
2. ConfigurationAdapter - Translates legacy configuration parameters to new modular configs
3. Helper functions for converting between legacy and new model representations
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlflow
import numpy as np
import pyro
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, Int
from numpy import ndarray
from scvi.data import AnnDataManager
from scvi.model.base import BaseModelClass

from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models._velocity_module import VelocityModule
from pyrovelocity.models.factory import (
    DynamicsModelConfig,
    InferenceGuideConfig,
    LikelihoodModelConfig,
    ObservationModelConfig,
    PriorModelConfig,
    PyroVelocityModelConfig,
    create_model,
)
from pyrovelocity.models.model import ModelState, PyroVelocityModel


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
        cell_specific_kinetics = legacy_params.get("cell_specific_kinetics", None)
        kinetics_num = legacy_params.get("kinetics_num", None)
        
        # Map legacy model_type to dynamics model configuration
        dynamics_params = {
            "shared_time": shared_time,
            "t_scale_on": t_scale_on,
            "cell_specific_kinetics": cell_specific_kinetics,
            "kinetics_num": kinetics_num,
        }
        
        # Map legacy likelihood to likelihood model configuration
        likelihood_name = "poisson" if likelihood == "Poisson" else "negative_binomial"
        
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
        legacy_params = modular_config.metadata.copy() if modular_config.metadata else {}
        
        # Map dynamics model configuration to legacy model_type
        dynamics_name = modular_config.dynamics_model.name
        legacy_params["model_type"] = "auto" if dynamics_name == "standard" else dynamics_name
        
        # Extract dynamics model parameters
        dynamics_params = modular_config.dynamics_model.params
        legacy_params["shared_time"] = dynamics_params.get("shared_time", True)
        legacy_params["t_scale_on"] = dynamics_params.get("t_scale_on", False)
        legacy_params["cell_specific_kinetics"] = dynamics_params.get("cell_specific_kinetics", None)
        legacy_params["kinetics_num"] = dynamics_params.get("kinetics_num", None)
        
        # Map likelihood model configuration to legacy likelihood
        likelihood_name = modular_config.likelihood_model.name
        legacy_params["likelihood"] = "Poisson" if likelihood_name == "poisson" else "NegativeBinomial"
        
        # Map inference guide configuration to legacy guide_type
        guide_name = modular_config.inference_guide.name
        legacy_params["guide_type"] = "auto" if guide_name == "auto" else guide_name
        
        return legacy_params


class LegacyModelAdapter(PyroVelocity):
    """
    Adapter that makes the new PyroVelocityModel compatible with the legacy PyroVelocity API.
    
    This adapter implements the legacy PyroVelocity interface but delegates to the new
    modular PyroVelocityModel internally, allowing existing code to work with the new
    architecture without modification.
    """
    
    @beartype
    def __init__(
        self,
        adata: AnnData,
        modular_model: Optional[PyroVelocityModel] = None,
        **kwargs
    ):
        """
        Initialize the LegacyModelAdapter.
        
        Args:
            adata: AnnData object containing the gene expression data
            modular_model: Optional pre-initialized PyroVelocityModel
            **kwargs: Additional keyword arguments for legacy PyroVelocity initialization
        """
        # Initialize the base class with minimal functionality
        # We'll override most methods to delegate to the modular model
        BaseModelClass.__init__(self, adata)
        
        # Store the AnnData object
        self.adata = adata
        
        # Store legacy initialization parameters
        self.init_params_ = self._get_init_params(locals())
        self.init_params_.update(kwargs)
        
        # Create or store the modular model
        if modular_model is not None:
            self._modular_model = modular_model
        else:
            # Convert legacy parameters to modular configuration
            modular_config = ConfigurationAdapter.legacy_to_modular_config(kwargs)
            self._modular_model = create_model(modular_config)
        
        # Set up legacy attributes for compatibility
        self.use_gpu = kwargs.get("use_gpu", "auto")
        self.cell_specific_kinetics = kwargs.get("cell_specific_kinetics", None)
        self.k = kwargs.get("kinetics_num", None)
        
        # Set up layers based on input_type
        input_type = kwargs.get("input_type", "raw")
        if input_type == "knn":
            self.layers = ["Mu", "Ms"]
        elif input_type == "raw_cpm":
            self.layers = ["unspliced", "spliced"]
        else:
            self.layers = ["raw_unspliced", "raw_spliced"]
        
        self.input_type = input_type
        
        # Create a minimal module for compatibility with legacy code
        # This is a placeholder that will be replaced with adapter methods
        self.module = self._create_legacy_module_adapter()
        
        self.num_cells = self.adata.n_obs
        self._model_summary_string = """
        RNA velocity Pyro model with parameters (via LegacyModelAdapter):
        """
    
    def _create_legacy_module_adapter(self) -> VelocityModule:
        """
        Create a minimal VelocityModule adapter for compatibility with legacy code.
        
        Returns:
            A VelocityModule instance that delegates to the modular model
        """
        # This is a placeholder implementation
        # In a real implementation, we would create a proper adapter for VelocityModule
        # that delegates to the modular model components
        return VelocityModule(
            self.adata.n_obs,
            self.adata.n_vars,
            model_type="auto",
            guide_type="auto",
            likelihood="Poisson",
        )
    
    def train(self, **kwargs):
        """
        Train the model using the new modular architecture.
        
        Args:
            **kwargs: Training parameters
        """
        # Here we would implement training using the modular model
        # This is a placeholder implementation
        pyro.enable_validation(True)
        # In a real implementation, we would delegate to the modular model's training logic
        # For now, we'll just log that training would happen
        print("Training would be delegated to the modular model")
    
    def generate_posterior_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        num_samples: int = 100,
    ) -> Dict[str, ndarray]:
        """
        Generate posterior samples using the modular model.
        
        Args:
            adata: Optional AnnData object
            indices: Optional sequence of indices
            batch_size: Optional batch size
            num_samples: Number of posterior samples
            
        Returns:
            Dictionary of posterior samples
        """
        # Here we would implement posterior sample generation using the modular model
        # This is a placeholder implementation
        # In a real implementation, we would delegate to the modular model
        return {}
    
    @classmethod
    def from_modular_model(
        cls,
        adata: AnnData,
        modular_model: PyroVelocityModel,
    ) -> "LegacyModelAdapter":
        """
        Create a LegacyModelAdapter from a PyroVelocityModel.
        
        Args:
            adata: AnnData object
            modular_model: PyroVelocityModel instance
            
        Returns:
            LegacyModelAdapter instance
        """
        return cls(adata, modular_model=modular_model)


class ModularModelAdapter:
    """
    Adapter that makes the legacy PyroVelocity model compatible with the new PyroVelocityModel API.
    
    This adapter implements the new PyroVelocityModel interface but delegates to the legacy
    PyroVelocity model internally, allowing gradual migration to the new architecture.
    """
    
    @beartype
    def __init__(
        self,
        legacy_model: PyroVelocity,
    ):
        """
        Initialize the ModularModelAdapter.
        
        Args:
            legacy_model: Legacy PyroVelocity model instance
        """
        self.legacy_model = legacy_model
        
        # Create a minimal state for compatibility with the new architecture
        self.state = ModelState(
            dynamics_state={},
            prior_state={},
            likelihood_state={},
            observation_state={},
            guide_state={},
            metadata={"legacy_model": True},
        )
    
    @property
    def name(self) -> str:
        """Return the model name."""
        return "ModularModelAdapter"
    
    def get_state(self) -> ModelState:
        """Return the current model state."""
        return self.state
    
    def with_state(self, state: ModelState) -> "ModularModelAdapter":
        """
        Create a new model instance with the given state.
        
        Args:
            state: New model state
            
        Returns:
            New ModularModelAdapter instance with the given state
        """
        adapter = ModularModelAdapter(self.legacy_model)
        adapter.state = state
        return adapter
    
    @beartype
    def forward(
        self,
        x: Float[Array, "batch_size n_features"],
        time_points: Float[Array, "n_times"],
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through the model, delegating to the legacy model.
        
        Args:
            x: Input data tensor
            time_points: Time points
            cell_state: Optional cell state
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of model outputs
        """
        # This is a placeholder implementation
        # In a real implementation, we would delegate to the legacy model
        return {}
    
    @beartype
    def guide(
        self,
        x: Float[Array, "batch_size n_features"],
        time_points: Float[Array, "n_times"],
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Inference guide for the model, delegating to the legacy model.
        
        Args:
            x: Input data tensor
            time_points: Time points
            cell_state: Optional cell state
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of guide outputs
        """
        # This is a placeholder implementation
        # In a real implementation, we would delegate to the legacy model's guide
        return {}


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
    modular_config = ConfigurationAdapter.legacy_to_modular_config(legacy_config)
    
    # Create modular model
    modular_model = create_model(modular_config)
    
    # In a real implementation, we would also transfer trained parameters
    # from the legacy model to the modular model
    
    return modular_model


@beartype
def convert_modular_to_legacy(
    modular_model: PyroVelocityModel,
    adata: AnnData,
) -> PyroVelocity:
    """
    Convert a new PyroVelocityModel to a legacy PyroVelocity model.
    
    Args:
        modular_model: PyroVelocityModel instance
        adata: AnnData object
        
    Returns:
        Legacy PyroVelocity model instance
    """
    return LegacyModelAdapter(adata, modular_model=modular_model)


# Export all public symbols
__all__ = [
    # Adapter classes
    "LegacyModelAdapter",
    "ModularModelAdapter",
    "ConfigurationAdapter",
    
    # Conversion functions
    "convert_legacy_to_modular",
    "convert_modular_to_legacy",
]