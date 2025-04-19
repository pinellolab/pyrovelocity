"""Tests for observation models."""

import numpy as np
import pytest
import torch
from anndata import AnnData

from pyrovelocity.models.components.observations import StandardObservationModel
from pyrovelocity.models.registry import observation_model_registry


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    n_cells = 10
    n_genes = 5

    # Create random data
    unspliced = np.random.negative_binomial(5, 0.5, size=(n_cells, n_genes))
    spliced = np.random.negative_binomial(10, 0.5, size=(n_cells, n_genes))

    # Create AnnData object
    adata = AnnData(X=spliced)
    adata.layers["unspliced"] = unspliced
    adata.layers["spliced"] = spliced

    return adata


def test_standard_observation_model_init():
    """Test initialization of StandardObservationModel."""
    model = StandardObservationModel()
    assert model.use_observed_lib_size is True
    assert model.transform_batch is True

    model = StandardObservationModel(
        use_observed_lib_size=False, transform_batch=False
    )
    assert model.use_observed_lib_size is False
    assert model.transform_batch is False


def test_standard_observation_model_setup(mock_adata):
    """Test setup_observation method of StandardObservationModel."""
    model = StandardObservationModel()

    # Call setup_observation
    params = model.setup_observation(
        adata=mock_adata,
        num_cells=10,
        num_genes=5,
        plate_size=10,
    )

    # Check that the parameters are correctly set
    assert "u_obs" in params
    assert "s_obs" in params
    assert "u_scale" in params
    assert "s_scale" in params

    # Check that the parameters have the correct shape
    assert params["u_obs"].shape == (10, 5)
    assert params["s_obs"].shape == (10, 5)
    assert params["u_scale"].shape == (10, 1)
    assert params["s_scale"].shape == (10, 1)

    # Check that the attributes are correctly set
    assert hasattr(model, "u_obs")
    assert hasattr(model, "s_obs")
    assert hasattr(model, "u_scale")
    assert hasattr(model, "s_scale")


def test_standard_observation_model_no_lib_size(mock_adata):
    """Test StandardObservationModel with use_observed_lib_size=False."""
    model = StandardObservationModel(use_observed_lib_size=False)

    # Call setup_observation
    params = model.setup_observation(
        adata=mock_adata,
        num_cells=10,
        num_genes=5,
        plate_size=10,
    )

    # Check that the scaling factors are all ones
    assert torch.allclose(params["u_scale"], torch.ones_like(params["u_scale"]))
    assert torch.allclose(params["s_scale"], torch.ones_like(params["s_scale"]))


def test_observation_model_registry():
    """Test that the observation model is correctly registered."""
    # Check that the model is registered
    assert "standard" in observation_model_registry.available_models()

    # Check that we can retrieve the model class
    model_cls = observation_model_registry.get("standard")
    assert model_cls is StandardObservationModel

    # Check that we can create an instance
    model = observation_model_registry.create("standard")
    assert isinstance(model, StandardObservationModel)
