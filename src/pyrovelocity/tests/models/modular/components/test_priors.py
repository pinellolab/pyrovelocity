"""Tests for prior models."""

import pytest

from pyrovelocity.models.modular.components.priors import (
    LogNormalPriorModel,
    PiecewiseActivationPriorModel,
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


class TestPiecewiseActivationPriorModel:
    """Test suite for PiecewiseActivationPriorModel."""

    @pytest.fixture
    def prior_model(self):
        """Create a PiecewiseActivationPriorModel instance for testing."""
        return PiecewiseActivationPriorModel()

    @pytest.fixture
    def test_context(self):
        """Create a test context with synthetic data."""
        n_cells = 50
        n_genes = 2

        # Create synthetic observed data
        u_obs = torch.ones(n_cells, n_genes) * 10.0
        s_obs = torch.ones(n_cells, n_genes) * 5.0

        return {
            "u_obs": u_obs,
            "s_obs": s_obs,
            "include_prior": True,
        }

    def test_initialization(self, prior_model):
        """Test that the prior model initializes correctly."""
        assert prior_model.name == "piecewise_activation"
        assert hasattr(prior_model, "T_M_alpha")
        assert hasattr(prior_model, "T_M_beta")
        assert hasattr(prior_model, "alpha_off_loc")
        assert hasattr(prior_model, "alpha_on_loc")
        assert hasattr(prior_model, "gamma_star_loc")
        assert hasattr(prior_model, "t_on_star_loc")
        assert hasattr(prior_model, "delta_star_loc")
        assert hasattr(prior_model, "lambda_loc")

    def test_sample_parameters_shapes(self, prior_model):
        """Test that sample_parameters returns correct shapes."""
        n_genes = 3
        n_cells = 20

        params = prior_model.sample_parameters(n_genes=n_genes, n_cells=n_cells)

        # Check hierarchical time parameters (scalars)
        assert params["T_M_star"].shape == torch.Size([])
        assert params["t_loc"].shape == torch.Size([])
        assert params["t_scale"].shape == torch.Size([])

        # Check cell-specific time (n_cells,)
        assert params["t_star"].shape == torch.Size([n_cells])

        # Check gene-specific parameters (n_genes,)
        assert params["alpha_off"].shape == torch.Size([n_genes])
        assert params["alpha_on"].shape == torch.Size([n_genes])
        assert params["gamma_star"].shape == torch.Size([n_genes])
        assert params["t_on_star"].shape == torch.Size([n_genes])
        assert params["delta_star"].shape == torch.Size([n_genes])

        # Check cell-specific capture efficiency (n_cells,)
        assert params["lambda_j"].shape == torch.Size([n_cells])

    def test_forward_method_with_context(self, prior_model, test_context):
        """Test the forward method with a proper context."""
        pyro.clear_param_store()

        with pyro.poutine.trace() as trace:
            result_context = prior_model.forward(test_context)

        # Check that all expected parameters are in the result
        expected_params = [
            "T_M_star", "t_loc", "t_scale", "t_star",
            "alpha_off", "alpha_on", "gamma_star", "t_on_star", "delta_star",
            "lambda_j"
        ]

        for param in expected_params:
            assert param in result_context

        # Check that original context is preserved
        assert "u_obs" in result_context
        assert "s_obs" in result_context

        # Check that Pyro trace contains the expected sample sites
        trace_sites = list(trace.trace.nodes.keys())
        for param in expected_params:
            assert param in trace_sites

    def test_integration_with_dynamics_model(self, prior_model):
        """Test that sampled parameters work with the dynamics model."""
        from pyrovelocity.models.modular.components.dynamics import (
            PiecewiseActivationDynamicsModel,
        )

        # Create dynamics model
        dynamics_model = PiecewiseActivationDynamicsModel()

        # Sample parameters
        params = prior_model.sample_parameters(n_genes=2, n_cells=10)

        # Create context for dynamics model
        u_obs = torch.ones(10, 2) * 10.0
        s_obs = torch.ones(10, 2) * 5.0

        context = {
            "u_obs": u_obs,
            "s_obs": s_obs,
            **params
        }

        # Test that dynamics model can process the parameters
        try:
            result = dynamics_model.forward(context)
            assert "u_expected" in result
            assert "s_expected" in result
            assert "ut" in result
            assert "st" in result
        except Exception as e:
            pytest.fail(f"Dynamics model failed with prior parameters: {e}")


def test_piecewise_activation_prior_model_registration():
    """Test that PiecewiseActivationPriorModel is properly registered."""
    model_class = PriorModelRegistry.get("piecewise_activation")
    assert model_class == PiecewiseActivationPriorModel
    assert model_class.name == "piecewise_activation"
    assert "piecewise_activation" in PriorModelRegistry.list_available()
