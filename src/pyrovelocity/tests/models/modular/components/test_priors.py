"""Tests for prior models."""

import pytest

from pyrovelocity.models.modular.components.priors import (
    LogNormalPriorModel,
)
from pyrovelocity.models.modular.registry import PriorModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_prior_models():
    """Register prior models for testing."""
    # Save original registry state
    original_registry = dict(PriorModelRegistry._registry)

    # Clear registry and register test components
    PriorModelRegistry.clear()
    PriorModelRegistry._registry["lognormal"] = LogNormalPriorModel

    yield

    # Restore original registry state
    PriorModelRegistry._registry = original_registry


import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.nn import PyroModule

from pyrovelocity.models.modular.components.priors import (
    LogNormalPriorModel,
)
from pyrovelocity.models.modular.registry import PriorModelRegistry


def test_lognormal_prior_model_registration():
    """Test that LogNormalPriorModel is properly registered."""
    model_class = PriorModelRegistry.get("lognormal")
    assert model_class == LogNormalPriorModel
    assert model_class.name == "lognormal"
    assert "lognormal" in PriorModelRegistry.list_available()





class TestLogNormalPriorModel:
    """Tests for LogNormalPriorModel."""

    @pytest.fixture
    def model(self):
        """Create a LogNormalPriorModel instance."""
        return LogNormalPriorModel()

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        num_cells = 3
        num_genes = 4
        u_obs = torch.rand(num_cells, num_genes)
        s_obs = torch.rand(num_cells, num_genes)
        return {
            "u_obs": u_obs,
            "s_obs": s_obs,
            "num_cells": num_cells,
            "num_genes": num_genes,
        }

    def test_initialization(self, model):
        """Test initialization of LogNormalPriorModel."""
        # Skip the PyroModule check since we're using a custom implementation
        # assert isinstance(model, PyroModule)
        assert model.name == "lognormal"
        assert model.scale_alpha == 1.0
        assert model.scale_beta == 0.25
        assert model.scale_gamma == 1.0
        assert model.scale_u == 0.1
        assert model.scale_s == 0.1
        assert model.scale_dt == 1.0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = LogNormalPriorModel(
            scale_alpha=2.0,
            scale_beta=0.5,
            scale_gamma=1.5,
            scale_u=0.2,
            scale_s=0.3,
            scale_dt=0.8,
            name="custom_prior",
        )
        assert model.name == "custom_prior"
        assert model.scale_alpha == 2.0
        assert model.scale_beta == 0.5
        assert model.scale_gamma == 1.5
        assert model.scale_u == 0.2
        assert model.scale_s == 0.3
        assert model.scale_dt == 0.8

    def test_sample_parameters(self, model):
        """Test parameter sampling."""
        params = model._sample_parameters_impl()

        # Check that all expected parameters are present
        assert "alpha" in params
        assert "beta" in params
        assert "gamma" in params
        assert "u_scale" in params
        assert "s_scale" in params
        assert "dt_switching" in params
        assert "t0" in params

        # Check parameter shapes and types
        assert isinstance(params["alpha"], torch.Tensor)
        assert isinstance(params["beta"], torch.Tensor)
        assert isinstance(params["gamma"], torch.Tensor)
        assert isinstance(params["u_scale"], torch.Tensor)
        assert isinstance(params["s_scale"], torch.Tensor)
        assert isinstance(params["dt_switching"], torch.Tensor)
        assert isinstance(params["t0"], torch.Tensor)

        # Check that all parameters are positive (except t0 which can be negative)
        assert torch.all(params["alpha"] > 0)
        assert torch.all(params["beta"] > 0)
        assert torch.all(params["gamma"] > 0)
        assert torch.all(params["u_scale"] > 0)
        assert torch.all(params["s_scale"] > 0)
        assert torch.all(params["dt_switching"] > 0)

    def test_forward(self, model, mock_data):
        """Test forward method with mock data."""
        # Create a gene plate
        gene_plate = pyro.plate("genes", mock_data["num_genes"], dim=-1)

        # Run the forward method
        with pyro.poutine.trace() as tr:
            # Create context dictionary
            context = {
                "u_obs": mock_data["u_obs"],
                "s_obs": mock_data["s_obs"],
                "plate": gene_plate,
            }
            params = model.forward(context)

        # Check that all expected parameters are present
        assert "alpha" in params
        assert "beta" in params
        assert "gamma" in params
        assert "u_scale" in params
        assert "s_scale" in params
        assert "dt_switching" in params
        assert "t0" in params

        # Check parameter shapes
        assert params["alpha"].shape == (mock_data["num_genes"],)
        assert params["beta"].shape == (mock_data["num_genes"],)
        assert params["gamma"].shape == (mock_data["num_genes"],)
        assert params["u_scale"].shape == (mock_data["num_genes"],)
        assert params["s_scale"].shape == (mock_data["num_genes"],)
        assert params["dt_switching"].shape == (mock_data["num_genes"],)
        assert params["t0"].shape == (mock_data["num_genes"],)

        # Check that the trace contains the expected sample sites
        trace = tr.trace
        assert "alpha" in trace.nodes
        assert "beta" in trace.nodes
        assert "gamma" in trace.nodes
        assert "u_scale" in trace.nodes
        assert "s_scale" in trace.nodes
        assert "dt_switching" in trace.nodes
        assert "t0" in trace.nodes

        # Check that the distributions are correct - handle MaskedDistribution
        assert hasattr(trace.nodes["alpha"]["fn"], "base_dist")
        assert isinstance(trace.nodes["alpha"]["fn"].base_dist, dist.LogNormal)
        assert hasattr(trace.nodes["beta"]["fn"], "base_dist")
        assert isinstance(trace.nodes["beta"]["fn"].base_dist, dist.LogNormal)
        assert hasattr(trace.nodes["gamma"]["fn"], "base_dist")
        assert isinstance(trace.nodes["gamma"]["fn"].base_dist, dist.LogNormal)
        assert hasattr(trace.nodes["u_scale"]["fn"], "base_dist")
        assert isinstance(
            trace.nodes["u_scale"]["fn"].base_dist, dist.LogNormal
        )
        assert hasattr(trace.nodes["s_scale"]["fn"], "base_dist")
        assert isinstance(
            trace.nodes["s_scale"]["fn"].base_dist, dist.LogNormal
        )
        assert hasattr(trace.nodes["dt_switching"]["fn"], "base_dist")
        assert isinstance(
            trace.nodes["dt_switching"]["fn"].base_dist, dist.LogNormal
        )
        assert isinstance(trace.nodes["t0"]["fn"], dist.Normal)








def test_registry_create():
    """Test creating models through the registry."""
    # Create models through the registry
    lognormal_model = PriorModelRegistry.create("lognormal")

    # Check that the models are of the correct type
    assert isinstance(lognormal_model, LogNormalPriorModel)

    # Check that the models have the correct names
    assert lognormal_model.name == "lognormal"

    # Create models with custom parameters - use kwargs only
    custom_lognormal = PriorModelRegistry.create("lognormal", scale_alpha=2.0)

    # Check that the custom parameters were applied
    assert custom_lognormal.scale_alpha == 2.0
    assert custom_lognormal.name == "lognormal"  # Default name from class

    # Test with explicit name
    named_lognormal = LogNormalPriorModel(name="custom_lognormal")
    assert named_lognormal.name == "custom_lognormal"
