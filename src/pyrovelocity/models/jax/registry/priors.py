"""
Registry for prior functions in PyroVelocity JAX/NumPyro implementation.

This module provides a registry for prior functions, allowing registration
and retrieval of prior functions by name.
"""

from typing import Callable, List, Optional

from pyrovelocity.models.jax.interfaces import validate_prior_function
from pyrovelocity.models.jax.registry.base import Registry

# Singleton registry instance
_PRIOR_REGISTRY = None


class PriorRegistry(Registry):
    """
    Registry for prior functions.
    
    This class provides a registry for prior functions, allowing registration
    and retrieval of prior functions by name.
    """
    
    def __init__(self):
        """Initialize the prior registry."""
        super().__init__("priors")
    
    def register(self, name: str, fn: Callable) -> Callable:
        """
        Register a prior function in the registry.
        
        Args:
            name: Name to register the function under
            fn: Prior function to register
            
        Returns:
            The registered function
            
        Raises:
            ValueError: If a function with the same name is already registered
            TypeError: If the function does not conform to the prior function interface
        """
        # Validate the function
        validate_prior_function(fn)
        
        return super().register(name, fn)


def get_prior_registry() -> PriorRegistry:
    """
    Get the prior registry.
    
    Returns:
        The prior registry
    """
    global _PRIOR_REGISTRY
    if _PRIOR_REGISTRY is None:
        _PRIOR_REGISTRY = PriorRegistry()
    
    return _PRIOR_REGISTRY


def register_prior(name: str, fn: Callable) -> Callable:
    """
    Register a prior function.
    
    Args:
        name: Name to register the function under
        fn: Prior function to register
        
    Returns:
        The registered function
        
    Raises:
        ValueError: If a function with the same name is already registered
        TypeError: If the function does not conform to the prior function interface
    """
    registry = get_prior_registry()
    return registry.register(name, fn)


def get_prior(name: str) -> Optional[Callable]:
    """
    Get a prior function by name.
    
    Args:
        name: Name of the prior function to get
        
    Returns:
        The registered prior function, or None if not found
    """
    registry = get_prior_registry()
    return registry.get(name)


def list_priors() -> List[str]:
    """
    List all registered prior function names.
    
    Returns:
        List of registered prior function names
    """
    registry = get_prior_registry()
    return registry.list()
