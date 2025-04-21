"""
Registry for guide factory functions in PyroVelocity JAX/NumPyro implementation.

This module provides a registry for guide factory functions, allowing registration
and retrieval of guide factory functions by name.
"""

from typing import Callable, List, Optional

from pyrovelocity.models.jax.interfaces import validate_guide_factory_function
from pyrovelocity.models.jax.registry.base import Registry

# Singleton registry instance
_GUIDE_REGISTRY = None


class GuideRegistry(Registry):
    """
    Registry for guide factory functions.
    
    This class provides a registry for guide factory functions, allowing registration
    and retrieval of guide factory functions by name.
    """
    
    def __init__(self):
        """Initialize the guide registry."""
        super().__init__("guides")
    
    def register(self, name: str, fn: Callable) -> Callable:
        """
        Register a guide factory function in the registry.
        
        Args:
            name: Name to register the function under
            fn: Guide factory function to register
            
        Returns:
            The registered function
            
        Raises:
            ValueError: If a function with the same name is already registered
            TypeError: If the function does not conform to the guide factory function interface
        """
        # Validate the function
        validate_guide_factory_function(fn)
        
        return super().register(name, fn)


def get_guide_registry() -> GuideRegistry:
    """
    Get the guide registry.
    
    Returns:
        The guide registry
    """
    global _GUIDE_REGISTRY
    if _GUIDE_REGISTRY is None:
        _GUIDE_REGISTRY = GuideRegistry()
    
    return _GUIDE_REGISTRY


def register_guide(name: str, fn: Callable) -> Callable:
    """
    Register a guide factory function.
    
    Args:
        name: Name to register the function under
        fn: Guide factory function to register
        
    Returns:
        The registered function
        
    Raises:
        ValueError: If a function with the same name is already registered
        TypeError: If the function does not conform to the guide factory function interface
    """
    registry = get_guide_registry()
    return registry.register(name, fn)


def get_guide(name: str) -> Optional[Callable]:
    """
    Get a guide factory function by name.
    
    Args:
        name: Name of the guide factory function to get
        
    Returns:
        The registered guide factory function, or None if not found
    """
    registry = get_guide_registry()
    return registry.get(name)


def list_guides() -> List[str]:
    """
    List all registered guide factory function names.
    
    Returns:
        List of registered guide factory function names
    """
    registry = get_guide_registry()
    return registry.list()
