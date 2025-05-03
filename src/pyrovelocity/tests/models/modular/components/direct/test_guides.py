"""Tests for Protocol-First inference guide implementations."""

import pytest
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoNormal

from pyrovelocity.models.modular.components.direct.guides import AutoGuideFactoryDirect
from pyrovelocity.models.modular.interfaces import InferenceGuide
from pyrovelocity.models.modular.registry import inference_guide_registry


@pytest.fixture(scope="module", autouse=True)
def register_guides():
    """Register inference guides for testing."""
    # Save original registry state
    original_registry = dict(inference_guide_registry._registry)
    
    # Clear registry and register test components
    inference_guide_registry.clear()
    inference_guide_registry._registry["auto_direct"] = AutoGuideFactoryDirect
    
    yield
    
    # Restore original registry state
    inference_guide_registry._registry = original_registry


@pytest.fixture
def simple_model():
    """Create a simple Pyro model for testing."""
    def model():
        x = pyro.sample("x", dist.Normal(0, 1))
        y = pyro.sample("y", dist.Normal(x, 1))
        return y
    
    return model


def test_auto_guide_factory_direct_registration():
    """Test that AutoGuideFactoryDirect is properly registered."""
    guide_class = inference_guide_registry.get("auto_direct")
    assert guide_class == AutoGuideFactoryDirect
    assert "auto_direct" in inference_guide_registry.list_available()


def test_auto_guide_factory_direct_initialization():
    """Test initialization of AutoGuideFactoryDirect."""
    guide_factory = AutoGuideFactoryDirect()
    assert guide_factory.name == "inference_guide_direct"
    assert guide_factory.guide_type == "AutoNormal"
    assert guide_factory.init_scale == 0.1
    
    guide_factory = AutoGuideFactoryDirect(
        name="custom_name",
        guide_type="AutoDelta",
        init_scale=0.5,
    )
    assert guide_factory.name == "custom_name"
    assert guide_factory.guide_type == "AutoDelta"
    assert guide_factory.init_scale == 0.5


def test_auto_guide_factory_direct_protocol():
    """Test that AutoGuideFactoryDirect implements the InferenceGuide Protocol."""
    guide_factory = AutoGuideFactoryDirect()
    assert isinstance(guide_factory, InferenceGuide)


def test_auto_guide_factory_direct_create_guide(simple_model):
    """Test create_guide method of AutoGuideFactoryDirect."""
    guide_factory = AutoGuideFactoryDirect()
    
    # Create guide
    guide = guide_factory.create_guide(simple_model)
    
    # Check that the guide is created
    assert guide is not None
    assert guide_factory._guide is not None
    assert isinstance(guide, AutoNormal)
    
    # Check that we can get the guide
    retrieved_guide = guide_factory.get_guide()
    assert retrieved_guide is guide


def test_auto_guide_factory_direct_invalid_type():
    """Test AutoGuideFactoryDirect with invalid guide type."""
    guide_factory = AutoGuideFactoryDirect(guide_type="InvalidType")
    
    # Creating guide should raise ValueError
    with pytest.raises(ValueError):
        guide_factory.create_guide(lambda: None)


def test_auto_guide_factory_direct_get_guide_without_create():
    """Test get_guide without creating guide first."""
    guide_factory = AutoGuideFactoryDirect()
    
    # Getting guide without creating it should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide_factory.get_guide()


def test_auto_guide_factory_direct_call(simple_model):
    """Test __call__ method of AutoGuideFactoryDirect."""
    guide_factory = AutoGuideFactoryDirect()
    
    # Call the guide factory with the model
    guide_fn = guide_factory(simple_model)
    
    # Check that the guide is created
    assert guide_factory._guide is not None
    assert guide_factory._model is simple_model
    
    # Check that the guide function is callable
    assert callable(guide_fn)


def test_auto_guide_factory_direct_sample_posterior(simple_model):
    """Test sample_posterior method of AutoGuideFactoryDirect."""
    guide_factory = AutoGuideFactoryDirect()
    
    # Create guide
    guide = guide_factory.create_guide(simple_model)
    
    # Initialize parameters
    pyro.clear_param_store()
    guide()
    
    # Sample from posterior
    samples = guide_factory.sample_posterior(num_samples=10)
    
    # Check that samples are generated
    assert "x" in samples
    assert "y" in samples
    assert samples["x"].shape[0] == 10
    assert samples["y"].shape[0] == 10
