"""
Base registry implementation for PyroVelocity JAX/NumPyro components.

This module provides a base registry implementation for registering and retrieving
functions for different component types in the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, List, Optional, Type
import functools

# Global registry of registries
_REGISTRIES: Dict[str, "Registry"] = {}


class Registry:
    """
    Registry for functions of a specific type.
    
    This class provides a registry for functions of a specific type, allowing
    registration and retrieval of functions by name.
    
    Attributes:
        name: Name of the registry
        _registry: Dictionary mapping function names to functions
    """
    
    def __init__(self, name: str):
        """
        Initialize a registry.
        
        Args:
            name: Name of the registry
        """
        self.name = name
        self._registry: Dict[str, Callable] = {}
        
        # Register this registry in the global registry
        _REGISTRIES[name] = self
    
    def register(self, name: str, fn: Callable) -> Callable:
        """
        Register a function in the registry.
        
        Args:
            name: Name to register the function under
            fn: Function to register
            
        Returns:
            The registered function
            
        Raises:
            ValueError: If a function with the same name is already registered
        """
        if name in self._registry:
            raise ValueError(f"Function '{name}' is already registered in registry '{self.name}'")
        
        self._registry[name] = fn
        return fn
    
    def get(self, name: str) -> Optional[Callable]:
        """
        Get a function from the registry.
        
        Args:
            name: Name of the function to get
            
        Returns:
            The registered function, or None if not found
        """
        return self._registry.get(name)
    
    def list(self) -> List[str]:
        """
        List all registered function names.
        
        Returns:
            List of registered function names
        """
        return list(self._registry.keys())


def get_registry(name: str) -> Registry:
    """
    Get a registry by name.
    
    Args:
        name: Name of the registry to get
        
    Returns:
        The registry with the given name
        
    Raises:
        ValueError: If no registry with the given name exists
    """
    if name not in _REGISTRIES:
        raise ValueError(f"No registry with name '{name}' exists")
    
    return _REGISTRIES[name]


def register(registry_name: str, fn_name: str) -> Callable:
    """
    Decorator for registering a function in a registry.
    
    Args:
        registry_name: Name of the registry to register in
        fn_name: Name to register the function under
        
    Returns:
        Decorator function
        
    Example:
        >>> @register("dynamics", "standard")
        >>> def standard_dynamics_function(...):
        >>>     ...
    """
    def decorator(fn: Callable) -> Callable:
        registry = get_registry(registry_name)
        registry.register(fn_name, fn)
        return fn
    
    return decorator
