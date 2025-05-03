"""Tests for Protocol-First likelihood model implementations."""

import pyro
import pytest
import torch

from pyrovelocity.models.modular.components.direct.likelihoods import (
    NegativeBinomialLikelihoodModelDirect,
    PoissonLikelihoodModelDirect,
)
from pyrovelocity.models.modular.interfaces import LikelihoodModel
from pyrovelocity.models.modular.registry import LikelihoodModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_likelihood_models():
    """Register likelihood models for testing."""
    # Save original registry state
    original_registry = dict(LikelihoodModelRegistry._registry)

    # Clear registry and register test components
    LikelihoodModelRegistry.clear()
    LikelihoodModelRegistry._registry["poisson_direct"] = PoissonLikelihoodModelDirect
    LikelihoodModelRegistry._registry["negative_binomial_direct"] = NegativeBinomialLikelihoodModelDirect

    yield

    # Restore original registry state
    LikelihoodModelRegistry._registry = original_registry


def test_poisson_likelihood_model_direct_registration():
    """Test that PoissonLikelihoodModelDirect is properly registered."""
    model_class = LikelihoodModelRegistry.get("poisson_direct")
    assert model_class == PoissonLikelihoodModelDirect
    assert "poisson_direct" in LikelihoodModelRegistry.list_available()


def test_poisson_likelihood_model_direct_initialization():
    """Test initialization of PoissonLikelihoodModelDirect."""
    model = PoissonLikelihoodModelDirect()
    assert model.name == "poisson_likelihood_direct"

    model = PoissonLikelihoodModelDirect(name="custom_name")
    assert model.name == "custom_name"


def test_poisson_likelihood_model_direct_protocol():
    """Test that PoissonLikelihoodModelDirect implements the LikelihoodModel Protocol."""
    model = PoissonLikelihoodModelDirect()
    assert isinstance(model, LikelihoodModel)


def test_poisson_likelihood_model_direct_forward():
    """Test forward method of PoissonLikelihoodModelDirect."""
    model = PoissonLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    u_expected = torch.rand(batch_size, n_genes)
    s_expected = torch.rand(batch_size, n_genes)

    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
    }

    # Initialize Pyro
    pyro.clear_param_store()

    # Call forward method
    with pyro.poutine.trace() as tr:
        result = model.forward(context)

    # Check that the context is updated with distributions
    assert "u_dist" in result
    assert "s_dist" in result

    # Check that the trace contains the expected observe sites
    trace = tr.trace
    assert "u_obs" in trace.nodes
    assert "s_obs" in trace.nodes
    assert trace.nodes["u_obs"]["type"] == "sample"
    assert trace.nodes["s_obs"]["type"] == "sample"
    assert trace.nodes["u_obs"]["is_observed"]
    assert trace.nodes["s_obs"]["is_observed"]


def test_poisson_likelihood_model_direct_log_prob():
    """Test log_prob method of PoissonLikelihoodModelDirect."""
    model = PoissonLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    observations = torch.randint(0, 10, (batch_size, n_genes)).float()
    predictions = torch.rand(batch_size, n_genes)

    # Call log_prob method
    log_probs = model.log_prob(observations, predictions)

    # Check that the log probabilities are computed
    assert log_probs.shape == (batch_size,)


def test_poisson_likelihood_model_direct_sample():
    """Test sample method of PoissonLikelihoodModelDirect."""
    model = PoissonLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    predictions = torch.rand(batch_size, n_genes)

    # Call sample method
    samples = model.sample(predictions)

    # Check that the samples are generated
    assert samples.shape == (batch_size, n_genes)


def test_poisson_likelihood_model_direct_with_scaling():
    """Test PoissonLikelihoodModelDirect with scaling factors."""
    model = PoissonLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    u_expected = torch.rand(batch_size, n_genes)
    s_expected = torch.rand(batch_size, n_genes)
    u_scale = torch.rand(batch_size, 1)
    s_scale = torch.rand(batch_size, 1)

    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
        "u_scale": u_scale,
        "s_scale": s_scale,
    }

    # Initialize Pyro
    pyro.clear_param_store()

    # Call forward method
    with pyro.poutine.trace() as tr:
        result = model.forward(context)

    # Check that the context is updated with distributions
    assert "u_dist" in result
    assert "s_dist" in result

    # Check that the trace contains the expected observe sites
    trace = tr.trace
    assert "u_obs" in trace.nodes
    assert "s_obs" in trace.nodes
    assert trace.nodes["u_obs"]["type"] == "sample"
    assert trace.nodes["s_obs"]["type"] == "sample"
    assert trace.nodes["u_obs"]["is_observed"]
    assert trace.nodes["s_obs"]["is_observed"]


# Tests for NegativeBinomialLikelihoodModelDirect

def test_negative_binomial_likelihood_model_direct_registration():
    """Test that NegativeBinomialLikelihoodModelDirect is properly registered."""
    model_class = LikelihoodModelRegistry.get("negative_binomial_direct")
    assert model_class == NegativeBinomialLikelihoodModelDirect
    assert "negative_binomial_direct" in LikelihoodModelRegistry.list_available()


def test_negative_binomial_likelihood_model_direct_initialization():
    """Test initialization of NegativeBinomialLikelihoodModelDirect."""
    model = NegativeBinomialLikelihoodModelDirect()
    assert model.name == "negative_binomial_likelihood_direct"

    model = NegativeBinomialLikelihoodModelDirect(name="custom_name")
    assert model.name == "custom_name"


def test_negative_binomial_likelihood_model_direct_protocol():
    """Test that NegativeBinomialLikelihoodModelDirect implements the LikelihoodModel Protocol."""
    model = NegativeBinomialLikelihoodModelDirect()
    assert isinstance(model, LikelihoodModel)


def test_negative_binomial_likelihood_model_direct_forward():
    """Test forward method of NegativeBinomialLikelihoodModelDirect."""
    model = NegativeBinomialLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    u_expected = torch.rand(batch_size, n_genes)
    s_expected = torch.rand(batch_size, n_genes)

    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
    }

    # Initialize Pyro
    pyro.clear_param_store()

    # Call forward method
    with pyro.poutine.trace() as tr:
        result = model.forward(context)

    # Check that the context is updated with distributions
    assert "u_dist" in result
    assert "s_dist" in result
    assert isinstance(result["u_dist"], pyro.distributions.NegativeBinomial)
    assert isinstance(result["s_dist"], pyro.distributions.NegativeBinomial)

    # Check that the trace contains the expected sample sites
    trace = tr.trace
    assert "u_obs" in trace.nodes
    assert "s_obs" in trace.nodes
    assert trace.nodes["u_obs"]["type"] == "sample"
    assert trace.nodes["s_obs"]["type"] == "sample"
    assert trace.nodes["u_obs"]["is_observed"]
    assert trace.nodes["s_obs"]["is_observed"]


def test_negative_binomial_likelihood_model_direct_log_prob():
    """Test log_prob method of NegativeBinomialLikelihoodModelDirect."""
    model = NegativeBinomialLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    observations = torch.randint(0, 10, (batch_size, n_genes)).float()
    predictions = torch.rand(batch_size, n_genes)

    # Call log_prob method
    log_probs = model.log_prob(observations, predictions)

    # Check that the log probabilities are computed
    assert log_probs.shape == (batch_size,)


def test_negative_binomial_likelihood_model_direct_sample():
    """Test sample method of NegativeBinomialLikelihoodModelDirect."""
    model = NegativeBinomialLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    predictions = torch.rand(batch_size, n_genes)

    # Call sample method
    samples = model.sample(predictions)

    # Check that the samples are generated
    assert samples.shape == (batch_size, n_genes)


def test_negative_binomial_likelihood_model_direct_with_scaling():
    """Test NegativeBinomialLikelihoodModelDirect with scaling factors."""
    model = NegativeBinomialLikelihoodModelDirect()

    # Create test data
    batch_size = 3
    n_genes = 4
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    u_expected = torch.rand(batch_size, n_genes)
    s_expected = torch.rand(batch_size, n_genes)
    u_scale = torch.rand(batch_size, 1)
    s_scale = torch.rand(batch_size, 1)

    # Create context
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
        "u_scale": u_scale,
        "s_scale": s_scale,
    }

    # Initialize Pyro
    pyro.clear_param_store()

    # Call forward method
    with pyro.poutine.trace() as tr:
        result = model.forward(context)

    # Check that the context is updated with distributions
    assert "u_dist" in result
    assert "s_dist" in result
    assert isinstance(result["u_dist"], pyro.distributions.NegativeBinomial)
    assert isinstance(result["s_dist"], pyro.distributions.NegativeBinomial)

    # Check that the trace contains the expected sample sites
    trace = tr.trace
    assert "u_obs" in trace.nodes
    assert "s_obs" in trace.nodes
    assert trace.nodes["u_obs"]["type"] == "sample"
    assert trace.nodes["s_obs"]["type"] == "sample"
    assert trace.nodes["u_obs"]["is_observed"]
    assert trace.nodes["s_obs"]["is_observed"]