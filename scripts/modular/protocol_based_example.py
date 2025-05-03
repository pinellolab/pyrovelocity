"""
Protocol-Based Architecture (now the default) Example for PyroVelocity's Modular Implementation.

This script demonstrates the Protocol-Based architecture in PyroVelocity's modular implementation,
which directly implements Protocol interfaces without inheriting from base classes.

The Protocol-Based approach (now the default) has several advantages:
1. Reduces code complexity by eliminating inheritance hierarchies
2. Enhances flexibility through Protocol interfaces
3. Creates perfect architectural consistency with the JAX implementation's pure functional approach
4. Allows for the discovery of natural abstractions through actual usage patterns
5. Avoids premature abstraction by initially allowing intentional duplication

This example shows:
1. Creating models using Protocol-Based components (now the default)
2. Side-by-side comparisons with base class implementations
3. Customizing Protocol-Based components (now the default)
4. Explaining the benefits of the Protocol-Based approach (now the default)
"""

import torch
import pyro
import matplotlib.pyplot as plt
import numpy as np

# Import Protocol interfaces
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    PriorModel,
    LikelihoodModel,
    ObservationModel,
    InferenceGuide,
)

# Import Protocol-Based components (now the default)
from pyrovelocity.models.modular.components.direct.dynamics import (
    StandardDynamicsModel,
    NonlinearDynamicsModel,
)
from pyrovelocity.models.modular.components.direct.priors import (
    LogNormalPriorModel,
    InformativePriorModel,
)
from pyrovelocity.models.modular.components.direct.likelihoods import (
    PoissonLikelihoodModel,
    NegativeBinomialLikelihoodModel,
)
from pyrovelocity.models.modular.components.direct.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.modular.components.direct.guides import (
    AutoGuideFactory,
    NormalGuide,
    DeltaGuide,
)

# Import base class implementations for comparison
from pyrovelocity.models.modular.components import (
    StandardDynamicsModel,
    LogNormalPriorModel,
    PoissonLikelihoodModel,
    StandardObservationModel,
    AutoGuideFactory,
)

# Import factory functions
from pyrovelocity.models.modular.factory import (
    create_standard_model,
    create_model,
    create_model_from_config,
)

# Import configuration classes
from pyrovelocity.models.modular.config import (
    ModelConfig,
    ComponentConfig,
)

# Import PyroVelocityModel
from pyrovelocity.models.modular.model import PyroVelocityModel

# Import registry
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    PriorModelRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    InferenceGuideRegistry,
)

# Import utility functions
from pyrovelocity.models.modular.utils.context_utils import validate_context


def generate_synthetic_data(n_cells=100, n_genes=10, batch_size=1):
    """Generate synthetic data for demonstration."""
    # Create synthetic data
    u_obs = torch.rand(batch_size, n_cells, n_genes) * 10
    s_obs = torch.rand(batch_size, n_cells, n_genes) * 15

    # Add some structure to the data
    alpha = torch.ones(n_genes) * 2.0
    beta = torch.ones(n_genes) * 0.5
    gamma = torch.ones(n_genes) * 0.3

    # Create a context dictionary
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }

    return context


def compare_base_and_protocol_first_dynamics():
    """Compare base class and Protocol-Based dynamics models."""
    print("\n=== Comparing Base Class and Protocol-Based Dynamics Models ===")

    # Create base class and Protocol-Based dynamics models
    base_dynamics = StandardDynamicsModel()
    protocol_dynamics = StandardDynamicsModel()

    # Generate test data
    n_genes = 3
    alpha = torch.ones(n_genes) * 2.0
    beta = torch.ones(n_genes) * 0.5
    gamma = torch.ones(n_genes) * 0.3

    # Compute steady state
    base_u_ss, base_s_ss = base_dynamics.steady_state(alpha, beta, gamma)
    protocol_u_ss, protocol_s_ss = protocol_dynamics.steady_state(alpha, beta, gamma)

    # Compare results
    print("Base class steady state:")
    print(f"  u_ss: {base_u_ss}")
    print(f"  s_ss: {base_s_ss}")

    print("Protocol-Based steady state:")
    print(f"  u_ss: {protocol_u_ss}")
    print(f"  s_ss: {protocol_s_ss}")

    # Check if results are identical
    u_equal = torch.allclose(base_u_ss, protocol_u_ss)
    s_equal = torch.allclose(base_s_ss, protocol_s_ss)
    print(f"Results are identical: {u_equal and s_equal}")

    # Create context for forward pass
    # Use 2D tensors (cells, genes) instead of 3D tensors (batch, cells, genes)
    u_obs = torch.ones(10, n_genes) * 4.0
    s_obs = torch.ones(10, n_genes) * 6.0
    t = torch.ones(10, 1) * 2.0

    # For base class implementation, we need to create a context dictionary
    base_context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "t": t,
    }

    # Call forward method with context dictionary
    base_result = base_dynamics.forward(base_context)
    u_expected = base_result["u_expected"]
    s_expected = base_result["s_expected"]

    # For Protocol-Based implementation (now the default), we use the context dictionary
    protocol_context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "t": t,
    }

    # Run forward pass
    protocol_result = protocol_dynamics.forward(protocol_context)

    # Compare forward results
    print("\nBase class forward results:")
    print(f"  u_expected shape: {u_expected.shape}")
    print(f"  s_expected shape: {s_expected.shape}")

    print("Protocol-Based forward results:")
    print(f"  u_expected shape: {protocol_result['u_expected'].shape}")
    print(f"  s_expected shape: {protocol_result['s_expected'].shape}")

    # Check if results are identical
    u_equal = torch.allclose(u_expected, protocol_result['u_expected'])
    s_equal = torch.allclose(s_expected, protocol_result['s_expected'])
    print(f"Forward results are identical: {u_equal and s_equal}")

    # Highlight the differences in implementation
    print("\nKey differences in implementation:")
    print("1. Base class inherits from BaseDynamicsModel, which provides common functionality")
    print("2. Protocol-Based implementation (now the default) directly implements the DynamicsModel Protocol")
    print("3. Protocol-Based implementation (now the default) uses utility functions for common functionality")
    print("4. Protocol-Based implementation (now the default) allows for more flexible composition")


def compare_base_and_protocol_first_models():
    """Compare complete models using base class and Protocol-Based components (now the default)."""
    print("\n=== Comparing Complete Models with Base Class and Protocol-Based Components ===")

    # Create models
    base_model = create_standard_model()
    protocol_model = create_model()

    # Print model components
    print("Base model components:")
    print(f"  Dynamics: {base_model.dynamics_model.__class__.__name__}")
    print(f"  Prior: {base_model.prior_model.__class__.__name__}")
    print(f"  Likelihood: {base_model.likelihood_model.__class__.__name__}")
    print(f"  Observation: {base_model.observation_model.__class__.__name__}")
    print(f"  Guide: {base_model.guide_model.__class__.__name__}")

    print("\nProtocol-Based model components:")
    print(f"  Dynamics: {protocol_model.dynamics_model.__class__.__name__}")
    print(f"  Prior: {protocol_model.prior_model.__class__.__name__}")
    print(f"  Likelihood: {protocol_model.likelihood_model.__class__.__name__}")
    print(f"  Observation: {protocol_model.observation_model.__class__.__name__}")
    print(f"  Guide: {protocol_model.guide_model.__class__.__name__}")

    # Generate synthetic data
    context = generate_synthetic_data()

    # Run forward pass
    try:
        base_result = base_model.forward(**context)
        protocol_result = protocol_model.forward(**context)

        print("\nBoth models ran successfully!")

        # Check if results have the same keys
        base_keys = set(base_result.keys())
        protocol_keys = set(protocol_result.keys())

        print(f"Base model result keys: {base_keys}")
        print(f"Protocol-Based model result keys: {protocol_keys}")

        # Check if results are similar (not necessarily identical due to random initialization)
        common_keys = base_keys.intersection(protocol_keys)
        for key in common_keys:
            if isinstance(base_result[key], torch.Tensor) and isinstance(protocol_result[key], torch.Tensor):
                try:
                    # Check if shapes are the same
                    shape_equal = base_result[key].shape == protocol_result[key].shape
                    print(f"  {key}: Shapes are equal: {shape_equal}")
                except:
                    print(f"  {key}: Could not compare shapes")
    except Exception as e:
        print(f"Error running models: {e}")


def create_custom_protocol_first_component():
    """Create a custom Protocol-Based component."""
    print("\n=== Creating a Custom Protocol-Based Component ===")

    # Define a custom Protocol-Based dynamics model
    @DynamicsModelRegistry.register("custom_protocol_first")
    class CustomProtocolFirstDynamics:
        """
        A custom Protocol-Based dynamics model.

        This model directly implements the DynamicsModel Protocol without
        inheriting from BaseDynamicsModel. It demonstrates how to create
        custom components in the Protocol-Based architecture.
        """

        def __init__(self, name="custom_protocol_first", scaling_factor=2.0):
            """Initialize the custom dynamics model."""
            self.name = name
            self.scaling_factor = scaling_factor

        def forward(self, context):
            """
            Compute expected unspliced and spliced counts.

            Args:
                context: Dictionary containing model context

            Returns:
                Updated context with expected counts
            """
            # Validate context
            validate_context(context, required_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"])

            # Extract parameters
            u = context["u_obs"]
            s = context["s_obs"]
            alpha = context["alpha"]
            beta = context["beta"]
            gamma = context["gamma"]

            # Compute steady state
            u_ss, s_ss = self.steady_state(alpha, beta, gamma)

            # Apply scaling factor
            u_expected = u * self.scaling_factor
            s_expected = s * self.scaling_factor

            # Update context
            context["u_expected"] = u_expected
            context["s_expected"] = s_expected
            context["u_steady_state"] = u_ss
            context["s_steady_state"] = s_ss

            return context

        def steady_state(self, alpha, beta, gamma, **kwargs):
            """
            Compute steady state values.

            Args:
                alpha: Transcription rate
                beta: Splicing rate
                gamma: Degradation rate
                **kwargs: Additional parameters

            Returns:
                Tuple of (u_ss, s_ss)
            """
            # Custom steady state calculation
            u_ss = alpha / beta * self.scaling_factor
            s_ss = alpha / gamma * self.scaling_factor

            return u_ss, s_ss

    # Create an instance of the custom component
    custom_dynamics = DynamicsModelRegistry.create("custom_protocol_first")

    # Test the custom component
    n_genes = 3
    alpha = torch.ones(n_genes) * 2.0
    beta = torch.ones(n_genes) * 0.5
    gamma = torch.ones(n_genes) * 0.3

    # Compute steady state
    u_ss, s_ss = custom_dynamics.steady_state(alpha, beta, gamma)

    print("Custom Protocol-Based component:")
    print(f"  Class: {custom_dynamics.__class__.__name__}")
    print(f"  Steady state u_ss: {u_ss}")
    print(f"  Steady state s_ss: {s_ss}")

    # Create a model with the custom component
    custom_config = ModelConfig(
        dynamics_model=ComponentConfig(name="custom_protocol_first"),
        prior_model=ComponentConfig(name="lognormal"),
        likelihood_model=ComponentConfig(name="poisson"),
        observation_model=ComponentConfig(name="standard"),
        inference_guide=ComponentConfig(name="auto"),
    )

    custom_model = create_model_from_config(custom_config)

    print("\nCreated model with custom Protocol-Based component:")
    print(f"  Dynamics: {custom_model.dynamics_model.__class__.__name__}")
    print(f"  Prior: {custom_model.prior_model.__class__.__name__}")
    print(f"  Likelihood: {custom_model.likelihood_model.__class__.__name__}")
    print(f"  Observation: {custom_model.observation_model.__class__.__name__}")
    print(f"  Guide: {custom_model.guide_model.__class__.__name__}")


def explain_protocol_first_benefits():
    """Explain the benefits of the Protocol-Based approach (now the default)."""
    print("\n=== Benefits of the Protocol-Based Approach ===")

    print("1. Reduced code complexity")
    print("   - No inheritance hierarchies")
    print("   - Simpler mental model")
    print("   - Easier to understand and maintain")

    print("\n2. Enhanced flexibility")
    print("   - Components can be composed in any way that satisfies the Protocol")
    print("   - No constraints from base class implementation")
    print("   - Easier to create custom components")

    print("\n3. Architectural consistency with JAX implementation")
    print("   - Both implementations follow a functional approach")
    print("   - JAX implementation uses pure functions")
    print("   - Protocol-Based implementation (now the default) uses stateful objects with functional interfaces")
    print("   - Consistent mental model across implementations")

    print("\n4. Discovery of natural abstractions")
    print("   - Intentional duplication allows patterns to emerge naturally")
    print("   - Utility functions can be extracted based on actual usage")
    print("   - Avoids premature abstraction")

    print("\n5. Practical implementation")
    print("   - Utility functions for critical shared functionality")
    print("   - Context validation, error handling, PyroBuffer functionality")
    print("   - Registry system for component discovery and creation")
    print("   - Factory functions for easy model creation")


def main():
    """Run the Protocol-Based architecture example."""
    # Set random seed for reproducibility
    pyro.set_rng_seed(42)
    torch.manual_seed(42)

    print("=== Protocol-Based Architecture (now the default) Example ===")
    print("This example demonstrates the Protocol-Based architecture in PyroVelocity's modular implementation.")

    # Compare base class and Protocol-Based dynamics models
    compare_base_and_protocol_first_dynamics()

    # Compare complete models
    compare_base_and_protocol_first_models()

    # Create a custom Protocol-Based component
    create_custom_protocol_first_component()

    # Explain the benefits of the Protocol-Based approach (now the default)
    explain_protocol_first_benefits()

    print("\n=== Example Completed ===")


if __name__ == "__main__":
    main()
