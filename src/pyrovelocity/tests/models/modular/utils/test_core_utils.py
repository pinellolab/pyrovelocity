"""Tests for core utilities in PyroVelocity's modular architecture."""

import pytest
from expression import Result

from pyrovelocity.models.modular.utils.core_utils import ComponentError, create_error


def test_component_error_initialization():
    """Test initialization of ComponentError."""
    error = ComponentError(
        component="TestComponent",
        operation="test_operation",
        message="Test error message",
        details={"param1": 42, "param2": "test"},
    )
    
    assert error.component == "TestComponent"
    assert error.operation == "test_operation"
    assert error.message == "Test error message"
    assert error.details == {"param1": 42, "param2": "test"}


def test_component_error_initialization_without_details():
    """Test initialization of ComponentError without details."""
    error = ComponentError(
        component="TestComponent",
        operation="test_operation",
        message="Test error message",
    )
    
    assert error.component == "TestComponent"
    assert error.operation == "test_operation"
    assert error.message == "Test error message"
    assert error.details == {}


def test_create_error():
    """Test create_error function."""
    result = create_error(
        component_name="TestComponent",
        operation="test_operation",
        message="Test error message",
        details={"param1": 42, "param2": "test"},
    )
    
    assert isinstance(result, Result)
    assert result.is_error()
    assert "TestComponent.test_operation: Test error message" in str(result.error)


def test_create_error_without_details():
    """Test create_error function without details."""
    result = create_error(
        component_name="TestComponent",
        operation="test_operation",
        message="Test error message",
    )
    
    assert isinstance(result, Result)
    assert result.is_error()
    assert "TestComponent.test_operation: Test error message" in str(result.error)
