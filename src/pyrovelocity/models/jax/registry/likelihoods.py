"""
Registry for likelihood functions in PyroVelocity JAX/NumPyro implementation.

This module provides a registry for likelihood functions, allowing registration
and retrieval of likelihood functions by name.
"""

from typing import Callable, List, Optional

from pyrovelocity.models.jax.interfaces import validate_likelihood_function
from pyrovelocity.models.jax.registry.base import Registry

# Singleton registry instance
_LIKELIHOOD_REGISTRY = None


class LikelihoodRegistry(Registry):
    """
    Registry for likelihood functions.
    
    This class provides a registry for likelihood functions, allowing registration
    and retrieval of likelihood functions by name.
    """
    
    def __init__(self):
        """Initialize the likelihood registry."""
        super().__init__("likelihoods")
    
    def register(self, name: str, fn: Callable) -> Callable:
        """
        Register a likelihood function in the registry.
        
        Args:
            name: Name to register the function under
            fn: Likelihood function to register
            
        Returns:
            The registered function
            
        Raises:
            ValueError: If a function with the same name is already registered
            TypeError: If the function does not conform to the likelihood function interface
        """
        # Validate the function
        validate_likelihood_function(fn)
        
        return super().register(name, fn)


def get_likelihood_registry() -> LikelihoodRegistry:
    """
    Get the likelihood registry.
    
    Returns:
        The likelihood registry
    """
    global _LIKELIHOOD_REGISTRY
    if _LIKELIHOOD_REGISTRY is None:
        _LIKELIHOOD_REGISTRY = LikelihoodRegistry()
    
    return _LIKELIHOOD_REGISTRY


def register_likelihood(name: str, fn: Callable) -> Callable:
    """
    Register a likelihood function.
    
    Args:
        name: Name to register the function under
        fn: Likelihood function to register
        
    Returns:
        The registered function
        
    Raises:
        ValueError: If a function with the same name is already registered
        TypeError: If the function does not conform to the likelihood function interface
    """
    registry = get_likelihood_registry()
    return registry.register(name, fn)


def get_likelihood(name: str) -> Optional[Callable]:
    """
    Get a likelihood function by name.
    
    Args:
        name: Name of the likelihood function to get
        
    Returns:
        The registered likelihood function, or None if not found
    """
    registry = get_likelihood_registry()
    return registry.get(name)


def list_likelihoods() -> List[str]:
    """
    List all registered likelihood function names.
    
    Returns:
        List of registered likelihood function names
    """
    registry = get_likelihood_registry()
    return registry.list()
