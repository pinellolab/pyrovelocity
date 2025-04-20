"""Tests for AnnData integration utilities."""

import pytest
import numpy as np
import jax.numpy as jnp
import anndata
from jaxtyping import Array, Float

from pyrovelocity.models.jax.data.anndata import (
    prepare_anndata,
    extract_layers,
    store_results,
    get_library_size,
)


@pytest.fixture
def mock_anndata():
    """Create a mock AnnData object for testing."""
    n_cells = 10
    n_genes = 5
    
    # Create random data
    spliced = np.random.poisson(5, size=(n_cells, n_genes))
    unspliced = np.random.poisson(3, size=(n_cells, n_genes))
    
    # Create AnnData object
    adata = anndata.AnnData(X=spliced)
    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    
    # Add some metadata
    adata.obs["cell_type"] = np.random.choice(["A", "B", "C"], size=n_cells)
    adata.var["gene_name"] = [f"gene_{i}" for i in range(n_genes)]
    
    return adata


def test_prepare_anndata(mock_anndata):
    """Test prepare_anndata function."""
    # Call the function
    data_dict = prepare_anndata(mock_anndata)
    
    # Check that the returned object is a dictionary
    assert isinstance(data_dict, dict)
    
    # Check that the dictionary contains the expected keys
    expected_keys = ["X_spliced", "X_unspliced", "cell_types", "gene_names"]
    for key in expected_keys:
        assert key in data_dict
    
    # Check that the arrays have the correct shape
    assert data_dict["X_spliced"].shape == mock_anndata.layers["spliced"].shape
    assert data_dict["X_unspliced"].shape == mock_anndata.layers["unspliced"].shape
    
    # Check that the arrays are JAX arrays
    assert isinstance(data_dict["X_spliced"], jnp.ndarray)
    assert isinstance(data_dict["X_unspliced"], jnp.ndarray)


def test_extract_layers(mock_anndata):
    """Test extract_layers function."""
    # Call the function
    u, s = extract_layers(mock_anndata)
    
    # Check that the returned objects are JAX arrays
    assert isinstance(u, jnp.ndarray)
    assert isinstance(s, jnp.ndarray)
    
    # Check that the arrays have the correct shape
    assert u.shape == mock_anndata.layers["unspliced"].shape
    assert s.shape == mock_anndata.layers["spliced"].shape
    
    # Check that the arrays contain the correct data
    np.testing.assert_allclose(u, jnp.array(mock_anndata.layers["unspliced"]))
    np.testing.assert_allclose(s, jnp.array(mock_anndata.layers["spliced"]))


def test_store_results(mock_anndata):
    """Test store_results function."""
    # Create some mock results
    n_cells, n_genes = mock_anndata.shape
    
    results = {
        "velocity": jnp.ones((n_cells, n_genes)),
        "alpha": jnp.ones(n_genes) * 2.0,
        "beta": jnp.ones(n_genes) * 0.5,
        "gamma": jnp.ones(n_genes) * 0.3,
        "switching": jnp.zeros(n_cells),
        "latent_time": jnp.linspace(0, 1, n_cells),
    }
    
    # Call the function
    adata_out = store_results(mock_anndata, results)
    
    # Check that the returned object is an AnnData object
    assert isinstance(adata_out, anndata.AnnData)
    
    # Check that the original AnnData object was not modified
    assert id(adata_out) != id(mock_anndata)
    
    # Check that the results were stored correctly
    assert "velocity_model_velocity" in adata_out.layers
    assert "velocity_model_alpha" in adata_out.var
    assert "velocity_model_beta" in adata_out.var
    assert "velocity_model_gamma" in adata_out.var
    assert "velocity_model_switching" in adata_out.obs
    assert "velocity_model_latent_time" in adata_out.obs
    
    # Check that the arrays contain the correct data
    np.testing.assert_allclose(adata_out.layers["velocity_model_velocity"], np.array(results["velocity"]))
    np.testing.assert_allclose(adata_out.var["velocity_model_alpha"], np.array(results["alpha"]))
    np.testing.assert_allclose(adata_out.var["velocity_model_beta"], np.array(results["beta"]))
    np.testing.assert_allclose(adata_out.var["velocity_model_gamma"], np.array(results["gamma"]))
    np.testing.assert_allclose(adata_out.obs["velocity_model_switching"], np.array(results["switching"]))
    np.testing.assert_allclose(adata_out.obs["velocity_model_latent_time"], np.array(results["latent_time"]))


def test_get_library_size(mock_anndata):
    """Test get_library_size function."""
    # Call the function
    u_lib_size, s_lib_size = get_library_size(mock_anndata)
    
    # Check that the returned objects are JAX arrays
    assert isinstance(u_lib_size, jnp.ndarray)
    assert isinstance(s_lib_size, jnp.ndarray)
    
    # Check that the arrays have the correct shape
    assert u_lib_size.shape == (mock_anndata.n_obs,)
    assert s_lib_size.shape == (mock_anndata.n_obs,)
    
    # Check that the arrays contain the correct data
    expected_u_lib_size = jnp.array(mock_anndata.layers["unspliced"].sum(axis=1)).flatten()
    expected_s_lib_size = jnp.array(mock_anndata.layers["spliced"].sum(axis=1)).flatten()
    
    np.testing.assert_allclose(u_lib_size, expected_u_lib_size)
    np.testing.assert_allclose(s_lib_size, expected_s_lib_size)


def test_prepare_anndata_with_custom_layers(mock_anndata):
    """Test prepare_anndata function with custom layer names."""
    # Rename the layers
    mock_anndata.layers["custom_spliced"] = mock_anndata.layers["spliced"].copy()
    mock_anndata.layers["custom_unspliced"] = mock_anndata.layers["unspliced"].copy()
    
    # Call the function with custom layer names
    data_dict = prepare_anndata(
        mock_anndata,
        spliced_layer="custom_spliced",
        unspliced_layer="custom_unspliced"
    )
    
    # Check that the arrays have the correct shape
    assert data_dict["X_spliced"].shape == mock_anndata.layers["custom_spliced"].shape
    assert data_dict["X_unspliced"].shape == mock_anndata.layers["custom_unspliced"].shape
    
    # Check that the arrays contain the correct data
    np.testing.assert_allclose(data_dict["X_unspliced"], jnp.array(mock_anndata.layers["custom_unspliced"]))
    np.testing.assert_allclose(data_dict["X_spliced"], jnp.array(mock_anndata.layers["custom_spliced"]))


def test_store_results_with_custom_model_name(mock_anndata):
    """Test store_results function with a custom model name."""
    # Create some mock results
    n_cells, n_genes = mock_anndata.shape
    
    results = {
        "velocity": jnp.ones((n_cells, n_genes)),
        "alpha": jnp.ones(n_genes) * 2.0,
    }
    
    # Call the function with a custom model name
    model_name = "custom_model"
    adata_out = store_results(mock_anndata, results, model_name=model_name)
    
    # Check that the results were stored with the custom model name
    assert f"{model_name}_velocity" in adata_out.layers
    assert f"{model_name}_alpha" in adata_out.var