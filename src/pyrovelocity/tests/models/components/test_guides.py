"""Tests for inference guides."""

import pytest
import pyro
import pyro.distributions as dist
import torch

from pyrovelocity.models.components.guides import (
    AutoGuideFactory,
    NormalGuide,
    DeltaGuide,
)
from pyrovelocity.models.registry import inference_guide_registry


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
    assert guide_factory.guide_type == "AutoNormal"
    assert guide_factory.init_loc_fn is None
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


def test_normal_guide_init():
    """Test initialization of NormalGuide."""
    guide = NormalGuide()
    assert guide.init_scale == 0.1

    guide = NormalGuide(init_scale=0.5)
    assert guide.init_scale == 0.5


def test_normal_guide_create_guide(simple_model):
    """Test create_guide method of NormalGuide."""
    guide = NormalGuide()

    # Create guide
    guide_fn = guide.create_guide(simple_model)

    # Check that the guide function is created
    assert callable(guide_fn)

    # Check that we can get the guide
    retrieved_guide_fn = guide.get_guide()
    assert callable(retrieved_guide_fn)


def test_normal_guide_get_guide_without_create():
    """Test get_guide without creating guide first."""
    guide = NormalGuide()

    # Getting guide without creating it should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide.get_guide()


def test_normal_guide_get_posterior_without_create():
    """Test get_posterior without creating guide first."""
    guide = NormalGuide()

    # Getting posterior without creating guide should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide.get_posterior()


def test_delta_guide_init():
    """Test initialization of DeltaGuide."""
    guide = DeltaGuide()
    assert guide.init_values == {}

    init_values = {"x": torch.tensor(0.0)}
    guide = DeltaGuide(init_values=init_values)
    assert guide.init_values == init_values


def test_delta_guide_create_guide(simple_model):
    """Test create_guide method of DeltaGuide."""
    guide = DeltaGuide()

    # Create guide
    guide_fn = guide.create_guide(simple_model)

    # Check that the guide function is created
    assert callable(guide_fn)

    # Check that we can get the guide
    retrieved_guide_fn = guide.get_guide()
    assert callable(retrieved_guide_fn)


def test_delta_guide_get_guide_without_create():
    """Test get_guide without creating guide first."""
    guide = DeltaGuide()

    # Getting guide without creating it should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide.get_guide()


def test_delta_guide_get_posterior_without_create():
    """Test get_posterior without creating guide first."""
    guide = DeltaGuide()

    # Getting posterior without creating guide should raise RuntimeError
    with pytest.raises(RuntimeError):
        guide.get_posterior()


def test_inference_guide_registry():
    """Test that the inference guides are correctly registered."""
    # Check that the guides are registered
    available_guides = inference_guide_registry.available_models()
    assert "auto" in available_guides
    assert "normal" in available_guides
    assert "delta" in available_guides

    # Check that we can retrieve the guide classes
    auto_cls = inference_guide_registry.get("auto")
    normal_cls = inference_guide_registry.get("normal")
    delta_cls = inference_guide_registry.get("delta")

    assert auto_cls is AutoGuideFactory
    assert normal_cls is NormalGuide
    assert delta_cls is DeltaGuide

    # Check that we can create instances
    auto_guide = inference_guide_registry.create("auto")
    normal_guide = inference_guide_registry.create("normal")
    delta_guide = inference_guide_registry.create("delta")

    assert isinstance(auto_guide, AutoGuideFactory)
    assert isinstance(normal_guide, NormalGuide)
    assert isinstance(delta_guide, DeltaGuide)
