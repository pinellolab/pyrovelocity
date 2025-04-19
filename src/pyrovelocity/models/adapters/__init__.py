"""
Adapter layer for PyroVelocity's modular architecture.

This package provides adapters for backward compatibility with the existing PyroVelocity API,
allowing seamless transition from the legacy monolithic architecture to the new modular
component-based architecture. The adapters implement the Adapter pattern to translate
between old-style function calls and new component-based calls.

The package includes:
1. LegacyModelAdapter - Adapts the new PyroVelocityModel to the legacy PyroVelocity API
2. ModularModelAdapter - Adapts the legacy PyroVelocity to the new PyroVelocityModel API
3. ConfigurationAdapter - Translates legacy configuration parameters to new modular configs
4. Helper functions for converting between legacy and new model representations
"""

from pyrovelocity.models.adapters.mono_to_modular import (
    ConfigurationAdapter,
    convert_legacy_to_modular,
)
from pyrovelocity.models.adapters.modular_to_mono import (
    LegacyModelAdapter,
    ModularModelAdapter,
    convert_modular_to_legacy,
)

__all__ = [
    # Adapter classes
    "LegacyModelAdapter",
    "ModularModelAdapter",
    "ConfigurationAdapter",
    # Conversion functions
    "convert_legacy_to_modular",
    "convert_modular_to_legacy",
]
