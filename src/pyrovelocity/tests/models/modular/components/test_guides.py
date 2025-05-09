"""Tests for inference guides."""

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer.autoguide import init_to_median

from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    LegacyAutoGuideFactory,
)
from pyrovelocity.models.modular.registry import inference_guide_registry


@pytest.fixture(scope="module", autouse=True)
def register_guides():
    """Register inference guides for testing."""
    # Save original registry state
    original_registry = dict(inference_guide_registry._registry)

    # Import the register_standard_components function
    from pyrovelocity.models.modular.registry import (
        register_standard_components,
    )

    # Register standard components
    register_standard_components()

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


def test_auto_guide_factory_init():
    """Test initialization of AutoGuideFactory."""
    guide_factory = AutoGuideFactory()
    assert guide_factory.guide_type == "AutoLowRankMultivariateNormal"
    assert guide_factory.init_loc_fn is init_to_median
    assert guide_factory.init_scale == 0.1

    guide_factory = AutoGuideFactory(
        guide_type="AutoDiagonalNormal",
        init_scale=0.5,
    )
    assert guide_factory.guide_type == "AutoDiagonalNormal"
    assert guide_factory.init_scale == 0.5


def test_auto_guide_factory_create_guide(simple_model):
    """Test create_guide method of AutoGuideFactory."""
    guide_factory = AutoGuideFactory()

    # Create guide
    guide = guide_factory.create_guide(simple_model)

    # Check that the guide is created
    assert guide is not None
    assert guide_factory._guide is not None

    # Check that we can get the guide
    retrieved_guide = guide_factory.get_guide()
    assert retrieved_guide is guide


def test_auto_guide_factory_invalid_type():
    """Test AutoGuideFactory with invalid guide type."""
    guide_factory = AutoGuideFactory(guide_type="InvalidType")

    # Creating guide should raise ValueError
    with pytest.raises(ValueError):
        guide_factory.create_guide(lambda: None)


def test_auto_guide_factory_get_guide_without_create():
    """Test get_guide without creating guide first."""
    guide_factory = AutoGuideFactory()

    # Getting guide without creating it should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide_factory.get_guide()


def test_auto_guide_factory_get_posterior_without_create():
    """Test get_posterior without creating guide first."""
    guide_factory = AutoGuideFactory()

    # Getting posterior without creating guide should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide_factory.get_posterior()


def test_legacy_auto_guide_factory_init():
    """Test initialization of LegacyAutoGuideFactory."""
    guide_factory = LegacyAutoGuideFactory()
    assert guide_factory.init_scale == 0.1
    assert guide_factory.add_offset == False

    guide_factory = LegacyAutoGuideFactory(
        init_scale=0.5,
        add_offset=True,
    )
    assert guide_factory.init_scale == 0.5
    assert guide_factory.add_offset == True


def test_legacy_auto_guide_factory_create_guide(simple_model):
    """Test create_guide method of LegacyAutoGuideFactory."""
    guide_factory = LegacyAutoGuideFactory()

    # Create guide
    guide = guide_factory.create_guide(simple_model)

    # Check that the guide is created
    assert guide is not None
    assert guide_factory._guide is not None

    # Check that we can get the guide
    retrieved_guide = guide_factory.get_guide()
    assert retrieved_guide is guide


def test_legacy_auto_guide_factory_get_guide_without_create():
    """Test get_guide without creating guide first."""
    guide_factory = LegacyAutoGuideFactory()

    # Getting guide without creating it should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide_factory.get_guide()


def test_legacy_auto_guide_factory_get_posterior_without_create():
    """Test get_posterior without creating guide first."""
    guide_factory = LegacyAutoGuideFactory()

    # Getting posterior without creating guide should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide_factory.get_posterior()


def test_inference_guide_registry():
    """Test that the inference guides are correctly registered."""
    # Clear the registry first to avoid test interference
    inference_guide_registry.clear()

    # Register the guides manually
    inference_guide_registry._registry["auto"] = AutoGuideFactory
    inference_guide_registry._registry["legacy_auto"] = LegacyAutoGuideFactory

    # Check that the guides are registered
    available_guides = inference_guide_registry.available_models()
    assert "auto" in available_guides
    assert "legacy_auto" in available_guides

    # Check that we can retrieve the guide classes
    auto_cls = inference_guide_registry.get("auto")
    legacy_auto_cls = inference_guide_registry.get("legacy_auto")

    assert auto_cls is AutoGuideFactory
    assert legacy_auto_cls is LegacyAutoGuideFactory

    # Check that we can create instances
    auto_guide = inference_guide_registry.create("auto")
    legacy_auto_guide = inference_guide_registry.create("legacy_auto")

    assert isinstance(auto_guide, AutoGuideFactory)
    assert isinstance(legacy_auto_guide, LegacyAutoGuideFactory)
