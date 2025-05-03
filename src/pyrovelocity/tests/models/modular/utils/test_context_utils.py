"""Tests for context validation utilities in PyroVelocity's modular architecture."""

import pytest
import torch
from expression import Result

from pyrovelocity.models.modular.utils.context_utils import validate_context


def test_validate_context_success():
    """Test validate_context with valid context."""
    context = {
        "u_obs": torch.randn(10, 5),
        "s_obs": torch.randn(10, 5),
        "alpha": torch.randn(5),
        "beta": torch.randn(5),
        "gamma": torch.randn(5),
    }
    
    result = validate_context(
        component_name="TestComponent",
        context=context,
        required_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
        tensor_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
    )
    
    assert result is context


def test_validate_context_missing_key():
    """Test validate_context with missing key."""
    context = {
        "u_obs": torch.randn(10, 5),
        "s_obs": torch.randn(10, 5),
        "alpha": torch.randn(5),
        "beta": torch.randn(5),
    }
    
    result = validate_context(
        component_name="TestComponent",
        context=context,
        required_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
        tensor_keys=["u_obs", "s_obs", "alpha", "beta"],
    )
    
    assert isinstance(result, Result)
    assert result.is_error()
    assert "Missing required key: gamma" in str(result.error)


def test_validate_context_wrong_type():
    """Test validate_context with wrong type."""
    context = {
        "u_obs": torch.randn(10, 5),
        "s_obs": torch.randn(10, 5),
        "alpha": 0.5,  # Not a tensor
        "beta": torch.randn(5),
        "gamma": torch.randn(5),
    }
    
    result = validate_context(
        component_name="TestComponent",
        context=context,
        required_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
        tensor_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
    )
    
    assert isinstance(result, Result)
    assert result.is_error()
    assert "Key alpha must be a torch.Tensor" in str(result.error)


def test_validate_context_no_required_keys():
    """Test validate_context without required keys."""
    context = {
        "u_obs": torch.randn(10, 5),
        "s_obs": torch.randn(10, 5),
    }
    
    result = validate_context(
        component_name="TestComponent",
        context=context,
    )
    
    assert result is context


def test_validate_context_no_tensor_keys():
    """Test validate_context without tensor keys."""
    context = {
        "u_obs": torch.randn(10, 5),
        "s_obs": torch.randn(10, 5),
        "alpha": 0.5,  # Not a tensor, but not checked
    }
    
    result = validate_context(
        component_name="TestComponent",
        context=context,
        required_keys=["u_obs", "s_obs", "alpha"],
    )
    
    assert result is context
