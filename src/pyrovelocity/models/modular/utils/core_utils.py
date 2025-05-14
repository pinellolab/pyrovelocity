"""
Core utilities for PyroVelocity's modular architecture.

This module provides error handling utilities that were previously part of
the BaseComponent class. These utilities are now available as standalone
functions that can be used by Protocol-First component implementations.

The utilities in this module are designed to be used by components that
directly implement Protocol interfaces without inheriting from base classes.
"""

from typing import Any, Dict, Optional

from expression import Result


class ComponentError:
    """
    Error information for component operations.

    This class represents errors that occur during component operations.
    It includes information about the component, operation, error message,
    and additional details. It is used with the Result type from the
    expression library to implement railway-oriented programming for
    error handling.

    Attributes:
        component: The name of the component where the error occurred
        operation: The operation that failed
        message: A descriptive error message
        details: Additional details about the error (optional)

    Examples:
        ```python
        # Create an error
        error = ComponentError(
            component="DynamicsModel",
            operation="forward",
            message="Invalid parameters",
            details={"alpha": alpha, "beta": beta}
        )

        # Return as a Result.Error
        return Result.Error(error)
        ```
    """

    def __init__(
        self,
        component: str,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a ComponentError.

        Args:
            component: The name of the component where the error occurred
            operation: The operation that failed
            message: A descriptive error message
            details: Additional details about the error (optional)
        """
        self.component = component
        self.operation = operation
        self.message = message
        self.details = details or {}


def create_error(
    component_name: str,
    operation: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Create an Error result for a component operation.

    This function creates a Result.Error containing a ComponentError with
    the specified information. It is used to implement railway-oriented
    programming for error handling in component operations.

    Args:
        component_name: The name of the component where the error occurred.
        operation: The name of the operation that failed.
        message: A descriptive error message.
        details: Optional additional error details.

    Returns:
        A Result.Error containing a ComponentError.

    Examples:
        ```python
        def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
            # Validate inputs
            if "u_obs" not in context:
                return create_error(
                    self.__class__.__name__,
                    "forward",
                    "Missing required parameter: u_obs",
                    {"provided_keys": list(context.keys())}
                )

            # Process data
            try:
                # Implementation details
                return Result.Ok(context)
            except Exception as e:
                return create_error(
                    self.__class__.__name__,
                    "forward",
                    f"Error processing data: {str(e)}",
                    {"context_keys": list(context.keys())}
                )
        ```
    """
    error = ComponentError(
        component=component_name,
        operation=operation,
        message=message,
        details=details or {},
    )

    error_message = f"{component_name}.{operation}: {message}"
    return Result.Error(error_message)
