"""Tests for Protocol-First observation model implementations."""

import numpy as np
import pytest
import torch
from anndata import AnnData

from pyrovelocity.models.modular.components.direct.observations import StandardObservationModelDirect
from pyrovelocity.models.modular.interfaces import ObservationModel
from pyrovelocity.models.modular.registry import observation_model_registry


@pytest.fixture(scope="module", autouse=True)
def register_observation_models():
    """Register observation models for testing."""
    # Save original registry state
    original_registry = dict(observation_model_registry._registry)
    
    # Clear registry and register test components
    observation_model_registry.clear()
    observation_model_registry._registry["standard_direct"] = StandardObservationModelDirect
    
    yield
    
    # Restore original registry state
    observation_model_registry._registry = original_registry


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    # Create random data
    n_cells = 10
    n_genes = 5
    X = np.random.randint(0, 10, (n_cells, n_genes))
    
    # Create AnnData object
    adata = AnnData(X=X)
    
    # Add layers
    adata.layers["unspliced"] = np.random.randint(0, 10, (n_cells, n_genes))
    adata.layers["spliced"] = np.random.randint(0, 10, (n_cells, n_genes))
    
    return adata


def test_standard_observation_model_direct_registration():
    """Test that StandardObservationModelDirect is properly registered."""
    model_class = observation_model_registry.get("standard_direct")
    assert model_class == StandardObservationModelDirect
    assert "standard_direct" in observation_model_registry.list_available()


def test_standard_observation_model_direct_initialization():
    """Test initialization of StandardObservationModelDirect."""
    model = StandardObservationModelDirect()
    assert model.name == "observation_model_direct"
    assert model.use_observed_lib_size is True
    assert model.transform_batch is True
    assert model.batch_size == 128
    
    model = StandardObservationModelDirect(
        name="custom_name",
        use_observed_lib_size=False,
        transform_batch=False,
        batch_size=64,
    )
    assert model.name == "custom_name"
    assert model.use_observed_lib_size is False
    assert model.transform_batch is False
    assert model.batch_size == 64


def test_standard_observation_model_direct_protocol():
    """Test that StandardObservationModelDirect implements the ObservationModel Protocol."""
    model = StandardObservationModelDirect()
    assert isinstance(model, ObservationModel)


def test_standard_observation_model_direct_forward():
    """Test forward method of StandardObservationModelDirect."""
    model = StandardObservationModelDirect()
    
    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.rand(batch_size, n_genes)
    s_obs = torch.rand(batch_size, n_genes)
    
    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }
    
    # Call forward method
    result = model.forward(context)
    
    # Check that the context is updated with transformed data
    assert "u_obs" in result
    assert "s_obs" in result
    assert "u_scale" in result
    assert "s_scale" in result
    
    # Check that the transformed data has the correct shape
    assert result["u_obs"].shape == (batch_size, n_genes)
    assert result["s_obs"].shape == (batch_size, n_genes)
    assert result["u_scale"].shape == (batch_size, 1)
    assert result["s_scale"].shape == (batch_size, 1)


def test_standard_observation_model_direct_setup_observation(mock_adata):
    """Test setup_observation method of StandardObservationModelDirect."""
    model = StandardObservationModelDirect()
    
    # Call setup_observation method
    params = model.setup_observation(
        adata=mock_adata,
        num_cells=10,
        num_genes=5,
        plate_size=10,
    )
    
    # Check that the parameters are set up
    assert "u_obs" in params
    assert "s_obs" in params
    assert "u_scale" in params
    assert "s_scale" in params
    
    # Check that the parameters have the correct shape
    assert params["u_obs"].shape == (10, 5)
    assert params["s_obs"].shape == (10, 5)
    assert params["u_scale"].shape == (10, 1)
    assert params["s_scale"].shape == (10, 1)
    
    # Check that the model attributes are set
    assert model.u_obs is not None
    assert model.s_obs is not None
    assert model.u_scale is not None
    assert model.s_scale is not None


def test_standard_observation_model_direct_prepare_data(mock_adata):
    """Test prepare_data method of StandardObservationModelDirect."""
    model = StandardObservationModelDirect()
    
    # Call prepare_data method
    data = model.prepare_data(adata=mock_adata)
    
    # Check that the data is prepared
    assert "u_obs" in data
    assert "s_obs" in data
    assert "u_scale" in data
    assert "s_scale" in data
    assert "cell_indices" in data
    
    # Check that the data has the correct shape
    assert data["u_obs"].shape == (10, 5)
    assert data["s_obs"].shape == (10, 5)
    assert data["u_scale"].shape == (10, 1)
    assert data["s_scale"].shape == (10, 1)
    assert data["cell_indices"].shape == (10,)


def test_standard_observation_model_direct_no_lib_size():
    """Test StandardObservationModelDirect with use_observed_lib_size=False."""
    model = StandardObservationModelDirect(use_observed_lib_size=False)
    
    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.rand(batch_size, n_genes)
    s_obs = torch.rand(batch_size, n_genes)
    
    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }
    
    # Call forward method
    result = model.forward(context)
    
    # Check that the scaling factors are all ones
    assert torch.allclose(result["u_scale"], torch.ones_like(result["u_scale"]))
    assert torch.allclose(result["s_scale"], torch.ones_like(result["s_scale"]))
