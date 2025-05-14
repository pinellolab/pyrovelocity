"""
Pyro utilities for PyroVelocity's modular architecture.

This module provides utilities for working with Pyro in the modular architecture.
These utilities were previously part of the PyroBufferMixin class and are now
available as standalone functions.

The utilities in this module are designed to be used by components that
directly implement Protocol interfaces without inheriting from base classes.
"""

import torch


def register_buffer(obj: object, name: str, tensor: torch.Tensor) -> None:
    """
    Register a buffer with an object.

    This function mimics PyroModule's register_buffer method by storing
    a tensor as an attribute of the object. It is used to store tensors
    that should be part of the model's state but are not parameters.

    Args:
        obj: The object to register the buffer with.
        name: The name to register the buffer under.
        tensor: The tensor to register.

    Examples:
        ```python
        class MyPriorModel:
            def __init__(self, alpha_loc: float = 0.0, alpha_scale: float = 1.0):
                self.name = "my_prior_model"
                
                # Register buffers for prior parameters
                register_buffer(self, "alpha_loc", torch.tensor(alpha_loc))
                register_buffer(self, "alpha_scale", torch.tensor(alpha_scale))
        ```
    """
    setattr(obj, name, tensor)
