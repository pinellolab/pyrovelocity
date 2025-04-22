"""
Tests for the PyroVelocity adapter layer.

This module contains tests for the adapter layer that provides backward compatibility
between the legacy PyroVelocity API and the new modular architecture.
"""

import os
import pytest
import numpy as np
from anndata import AnnData

from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.modular.model import PyroVelocityModel, ModelState
from pyrovelocity.models.adapters import (
    ConfigurationAdapter,
    LegacyModelAdapter,
    ModularModelAdapter,
    convert_legacy_to_modular,
    convert_modular_to_legacy,
)
from pyrovelocity.models.modular.factory import (
    create_standard_model,
    standard_model_config,
    PyroVelocityModelConfig,
)


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    # Create a small AnnData object with random data
    n_obs = 10
    n_vars = 5

    # Create random count data
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))

    # Create AnnData object
    adata = AnnData(X)

    # Add spliced and unspliced layers
    adata.layers["spliced"] = np.random.negative_binomial(
        5, 0.3, size=(n_obs, n_vars)
    )
    adata.layers["unspliced"] = np.random.negative_binomial(
        3, 0.3, size=(n_obs, n_vars)
    )

    # Add raw layers
    adata.layers["raw_spliced"] = adata.layers["spliced"].copy()
    adata.layers["raw_unspliced"] = adata.layers["unspliced"].copy()

    # Add library size information
    adata.obs["u_lib_size_raw"] = adata.layers["unspliced"].sum(axis=1)
    adata.obs["s_lib_size_raw"] = adata.layers["spliced"].sum(axis=1)

    # Set up AnnData for PyroVelocity
    PyroVelocity.setup_anndata(adata)

    return adata


class TestConfigurationAdapter:
    """Tests for the ConfigurationAdapter class."""

    def test_legacy_to_modular_config(self):
        """Test conversion from legacy parameters to modular configuration."""
        # Define legacy parameters
        legacy_params = {
            "model_type": "auto",
            "guide_type": "auto",
            "likelihood": "Poisson",
            "shared_time": True,
            "t_scale_on": False,
            "cell_specific_kinetics": None,
            "kinetics_num": None,
        }

        # Convert to modular configuration
        modular_config = ConfigurationAdapter.legacy_to_modular_config(
            legacy_params
        )

        # Verify the conversion
        assert isinstance(modular_config, PyroVelocityModelConfig)
        assert modular_config.dynamics_model.name == "standard"
        assert modular_config.prior_model.name == "lognormal"
        assert modular_config.likelihood_model.name == "poisson"
        assert modular_config.observation_model.name == "standard"
        assert modular_config.inference_guide.name == "auto"
        assert modular_config.dynamics_model.params["shared_time"] == True
        assert modular_config.dynamics_model.params["t_scale_on"] == False
        assert (
            modular_config.dynamics_model.params["cell_specific_kinetics"]
            is None
        )
        assert modular_config.dynamics_model.params["kinetics_num"] is None

    def test_modular_to_legacy_config(self):
        """Test conversion from modular configuration to legacy parameters."""
        # Create a standard modular configuration
        modular_config = standard_model_config()

        # Add some dynamics parameters
        modular_config.dynamics_model.params.update(
            {
                "shared_time": True,
                "t_scale_on": False,
                "cell_specific_kinetics": None,
                "kinetics_num": None,
            }
        )

        # Convert to legacy parameters
        legacy_params = ConfigurationAdapter.modular_to_legacy_config(
            modular_config
        )

        # Verify the conversion
        assert legacy_params["model_type"] == "auto"
        assert legacy_params["guide_type"] == "auto"
        assert legacy_params["likelihood"] == "Poisson"
        assert legacy_params["shared_time"] == True
        assert legacy_params["t_scale_on"] == False
        assert legacy_params["cell_specific_kinetics"] is None
        assert legacy_params["kinetics_num"] is None


class TestLegacyModelAdapter:
    """Tests for the LegacyModelAdapter class."""

    def test_initialization(self, sample_adata):
        """Test initialization of LegacyModelAdapter."""
        # Set up AnnData for LegacyModelAdapter
        LegacyModelAdapter.setup_anndata(sample_adata)

        # Create a LegacyModelAdapter
        adapter = LegacyModelAdapter(sample_adata)

        # Verify the adapter
        assert isinstance(adapter, PyroVelocity)
        assert hasattr(adapter, "_modular_model")
        assert isinstance(adapter._modular_model, PyroVelocityModel)

    def test_from_modular_model(self, sample_adata):
        """Test creation of LegacyModelAdapter from a PyroVelocityModel."""
        # Set up AnnData for LegacyModelAdapter
        LegacyModelAdapter.setup_anndata(sample_adata)

        # Create a standard modular model
        modular_model = create_standard_model()

        # Create a LegacyModelAdapter from the modular model
        adapter = LegacyModelAdapter.from_modular_model(
            sample_adata, modular_model
        )

        # Verify the adapter
        assert isinstance(adapter, PyroVelocity)
        assert adapter._modular_model is modular_model


class TestModularModelAdapter:
    """Tests for the ModularModelAdapter class."""

    def test_initialization(self, sample_adata):
        """Test initialization of ModularModelAdapter."""
        # Create a legacy model
        legacy_model = PyroVelocity(sample_adata)

        # Create a ModularModelAdapter
        adapter = ModularModelAdapter(legacy_model)

        # Verify the adapter
        assert hasattr(adapter, "legacy_model")
        assert adapter.legacy_model is legacy_model
        assert hasattr(adapter, "state")
        assert isinstance(adapter.state, ModelState)

    def test_with_state(self, sample_adata):
        """Test the with_state method of ModularModelAdapter."""
        # Create a legacy model
        legacy_model = PyroVelocity(sample_adata)

        # Create a ModularModelAdapter
        adapter = ModularModelAdapter(legacy_model)

        # Create a new state
        new_state = ModelState(
            dynamics_state={"test": "value"},
            prior_state={},
            likelihood_state={},
            observation_state={},
            guide_state={},
            metadata={"new": True},
        )

        # Create a new adapter with the new state
        new_adapter = adapter.with_state(new_state)

        # Verify the new adapter
        assert new_adapter is not adapter
        assert new_adapter.legacy_model is legacy_model
        assert new_adapter.state is not adapter.state
        assert new_adapter.state.dynamics_state == {"test": "value"}
        assert new_adapter.state.metadata == {"new": True}


class TestConversionFunctions:
    """Tests for the conversion functions."""

    def test_convert_legacy_to_modular(self, sample_adata):
        """Test conversion from legacy PyroVelocity to PyroVelocityModel."""
        # Create a legacy model
        legacy_model = PyroVelocity(sample_adata)

        # Convert to modular model
        modular_model = convert_legacy_to_modular(legacy_model)

        # Verify the conversion
        assert isinstance(modular_model, PyroVelocityModel)

    def test_convert_modular_to_legacy(self, sample_adata):
        """Test conversion from PyroVelocityModel to legacy PyroVelocity."""
        # Set up AnnData for LegacyModelAdapter
        LegacyModelAdapter.setup_anndata(sample_adata)

        # Create a modular model
        modular_model = create_standard_model()

        # Convert to legacy model
        legacy_model = convert_modular_to_legacy(modular_model, sample_adata)

        # Verify the conversion
        assert isinstance(legacy_model, PyroVelocity)
        assert isinstance(legacy_model, LegacyModelAdapter)
        assert legacy_model._modular_model is modular_model


class TestEndToEndCompatibility:
    """End-to-end tests for adapter compatibility."""

    def test_legacy_api_with_modular_model(self, sample_adata, tmp_path):
        """Test using the legacy API with a modular model under the hood."""
        # Set up AnnData for LegacyModelAdapter
        LegacyModelAdapter.setup_anndata(sample_adata)

        # Create a LegacyModelAdapter
        adapter = LegacyModelAdapter(sample_adata)

        # Verify that it behaves like a PyroVelocity model
        assert isinstance(adapter, PyroVelocity)

        # Test that we can call legacy methods
        # Note: We're not actually training here, just verifying the method exists
        assert hasattr(adapter, "train")
        assert hasattr(adapter, "generate_posterior_samples")

        # Test saving and loading
        model_path = os.path.join(tmp_path, "test_model")
        os.makedirs(model_path, exist_ok=True)

        # Verify save method exists (not actually saving)
        assert hasattr(adapter, "save_model")

    def test_modular_api_with_legacy_model(self, sample_adata):
        """Test using the modular API with a legacy model under the hood."""
        # Create a legacy model
        legacy_model = PyroVelocity(sample_adata)

        # Create a ModularModelAdapter
        adapter = ModularModelAdapter(legacy_model)

        # Verify that it has the modular model interface
        assert hasattr(adapter, "forward")
        assert hasattr(adapter, "guide")
        assert hasattr(adapter, "get_state")
        assert hasattr(adapter, "with_state")

        # Test that we can call modular methods
        # Note: We're not actually running inference here, just verifying the methods exist
        assert callable(adapter.forward)
        assert callable(adapter.guide)
