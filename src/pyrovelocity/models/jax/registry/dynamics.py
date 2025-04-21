"""
Registry for dynamics functions in PyroVelocity JAX/NumPyro implementation.

This module provides a registry for dynamics functions, allowing registration
and retrieval of dynamics functions by name.
"""

from typing import Callable, List, Optional

from pyrovelocity.models.jax.interfaces import validate_dynamics_function
from pyrovelocity.models.jax.registry.base import Registry

# Singleton registry instance
_DYNAMICS_REGISTRY = None


class DynamicsRegistry(Registry):
    """
    Registry for dynamics functions.
    
    This class provides a registry for dynamics functions, allowing registration
    and retrieval of dynamics functions by name.
    """
    
    def __init__(self):
        """Initialize the dynamics registry."""
        super().__init__("dynamics")
    
    def register(self, name: str, fn: Callable) -> Callable:
        """
        Register a dynamics function in the registry.
        
        Args:
            name: Name to register the function under
            fn: Dynamics function to register
            
        Returns:
            The registered function
            
        Raises:
            ValueError: If a function with the same name is already registered
            TypeError: If the function does not conform to the dynamics function interface
        """
        # Validate the function
        validate_dynamics_function(fn)
        
        return super().register(name, fn)


def get_dynamics_registry() -> DynamicsRegistry:
    """
    Get the dynamics registry.
    
    Returns:
        The dynamics registry
    """
    global _DYNAMICS_REGISTRY
    if _DYNAMICS_REGISTRY is None:
        _DYNAMICS_REGISTRY = DynamicsRegistry()
    
    return _DYNAMICS_REGISTRY


def register_dynamics(name: str, fn: Callable) -> Callable:
    """
    Register a dynamics function.
    
    Args:
        name: Name to register the function under
        fn: Dynamics function to register
        
    Returns:
        The registered function
        
    Raises:
        ValueError: If a function with the same name is already registered
        TypeError: If the function does not conform to the dynamics function interface
    """
    registry = get_dynamics_registry()
    return registry.register(name, fn)


def get_dynamics(name: str) -> Optional[Callable]:
    """
    Get a dynamics function by name.
    
    Args:
        name: Name of the dynamics function to get
        
    Returns:
        The registered dynamics function, or None if not found
    """
    registry = get_dynamics_registry()
    return registry.get(name)


def list_dynamics() -> List[str]:
    """
    List all registered dynamics function names.
    
    Returns:
        List of registered dynamics function names
    """
    registry = get_dynamics_registry()
    return registry.list()
