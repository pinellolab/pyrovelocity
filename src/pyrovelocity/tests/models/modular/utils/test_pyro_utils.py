"""Tests for Pyro utilities in PyroVelocity's modular architecture."""

import pytest
import torch

from pyrovelocity.models.modular.utils.pyro_utils import register_buffer


def test_register_buffer():
    """Test register_buffer function."""
    # Create a simple object
    class TestObject:
        pass
    
    obj = TestObject()
    
    # Register a buffer
    tensor = torch.randn(5)
    register_buffer(obj, "test_buffer", tensor)
    
    # Check that the buffer is registered
    assert hasattr(obj, "test_buffer")
    assert obj.test_buffer is tensor
    assert torch.all(obj.test_buffer == tensor)


def test_register_buffer_multiple():
    """Test register_buffer function with multiple buffers."""
    # Create a simple object
    class TestObject:
        pass
    
    obj = TestObject()
    
    # Register multiple buffers
    tensor1 = torch.randn(5)
    tensor2 = torch.randn(3, 4)
    register_buffer(obj, "buffer1", tensor1)
    register_buffer(obj, "buffer2", tensor2)
    
    # Check that the buffers are registered
    assert hasattr(obj, "buffer1")
    assert hasattr(obj, "buffer2")
    assert obj.buffer1 is tensor1
    assert obj.buffer2 is tensor2
    assert torch.all(obj.buffer1 == tensor1)
    assert torch.all(obj.buffer2 == tensor2)


def test_register_buffer_overwrite():
    """Test register_buffer function overwriting existing attribute."""
    # Create a simple object
    class TestObject:
        def __init__(self):
            self.existing_attr = "original value"
    
    obj = TestObject()
    
    # Register a buffer with the same name as an existing attribute
    tensor = torch.randn(5)
    register_buffer(obj, "existing_attr", tensor)
    
    # Check that the attribute is overwritten
    assert hasattr(obj, "existing_attr")
    assert obj.existing_attr is tensor
    assert torch.all(obj.existing_attr == tensor)
