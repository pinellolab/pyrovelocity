"""
Protocol-Based Architecture Example for PyroVelocity's Modular Implementation.

This script demonstrates the Protocol-Based architecture in PyroVelocity's modular implementation,
which directly implements Protocol interfaces without using inheritance.

The Protocol-Based approach has several advantages:
1. Reduces code complexity by eliminating inheritance hierarchies
2. Enhances flexibility through Protocol interfaces
3. Creates perfect architectural consistency with the JAX implementation's pure functional approach
4. Allows for the discovery of natural abstractions through actual usage patterns
5. Avoids premature abstraction by initially allowing intentional duplication

This example shows:
1. Creating models using Protocol-Based components
2. Creating custom Protocol-Based components
3. Explaining the benefits of the Protocol-Based approach
"""

import torch
import pyro

# Import Protocol interfaces
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    PriorModel,
    LikelihoodModel,
    ObservationModel,
    InferenceGuide,
)

# Import Protocol-Based components (the default architecture)
from pyrovelocity.models.modular.components import (
    StandardDynamicsModel,
    NonlinearDynamicsModel,
    LogNormalPriorModel,
    InformativePriorModel,
    PoissonLikelihoodModel,
    NegativeBinomialLikelihoodModel,
    StandardObservationModel,
    AutoGuideFactory,
    NormalGuide,
    DeltaGuide,
)

# Import factory functions
from pyrovelocity.models.modular.factory import (
    create_standard_model,
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


def demonstrate_dynamics_model():
    """Demonstrate the Protocol-Based dynamics model."""
    print("\n=== Demonstrating Protocol-Based Dynamics Model ===")

    # Create dynamics model
    dynamics = StandardDynamicsModel()

    # Generate test data
    n_genes = 3
    alpha = torch.ones(n_genes) * 2.0
    beta = torch.ones(n_genes) * 0.5
    gamma = torch.ones(n_genes) * 0.3

    # Compute steady state
    u_ss, s_ss = dynamics.steady_state(alpha, beta, gamma)

    # Display results
    print("Steady state values:")
    print(f"  u_ss: {u_ss}")
    print(f"  s_ss: {s_ss}")

    # Create context for forward pass
    # Use 2D tensors (cells, genes) instead of 3D tensors (batch, cells, genes)
    u_obs = torch.ones(10, n_genes) * 4.0
    s_obs = torch.ones(10, n_genes) * 6.0
    t = torch.ones(10, 1) * 2.0

    # Create context dictionary
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "t": t,
    }

    # Run forward pass
    result = dynamics.forward(context)
    u_expected = result["u_expected"]
    s_expected = result["s_expected"]

    # Display forward results
    print("\nForward pass results:")
    print(f"  u_expected shape: {u_expected.shape}")
    print(f"  s_expected shape: {s_expected.shape}")

    # Highlight the key aspects of the implementation
    print("\nKey aspects of the Protocol-Based implementation:")
    print("1. Directly implements the DynamicsModel Protocol without inheritance")
    print("2. Uses utility functions for common functionality like context validation")
    print("3. Communicates through a context dictionary that is passed between components")
    print("4. Allows for flexible composition with other components")


def demonstrate_standard_model():
    """Demonstrate a standard model using Protocol-Based components."""
    print("\n=== Demonstrating Standard Model with Protocol-Based Components ===")

    # Create model
    model = create_standard_model()

    # Print model components
    print("Standard model components:")
    print(f"  Dynamics: {model.dynamics_model.__class__.__name__}")
    print(f"  Prior: {model.prior_model.__class__.__name__}")
    print(f"  Likelihood: {model.likelihood_model.__class__.__name__}")
    print(f"  Observation: {model.observation_model.__class__.__name__}")
    print(f"  Guide: {model.guide_model.__class__.__name__}")

    # Generate synthetic data
    context = generate_synthetic_data()

    # Run forward pass
    try:
        result = model.forward(**context)
        print("\nModel ran successfully!")

        # Display result keys
        result_keys = set(result.keys())
        print(f"Model result keys: {result_keys}")

        # Display shapes of key tensors
        for key in result_keys:
            if isinstance(result[key], torch.Tensor):
                try:
                    print(f"  {key}: Shape = {result[key].shape}")
                except:
                    print(f"  {key}: Could not get shape")
    except Exception as e:
        print(f"Error running model: {e}")


def create_custom_protocol_first_component():
    """Create a custom Protocol-Based component."""
    print("\n=== Creating a Custom Protocol-Based Component ===")

    # Define a custom Protocol-Based dynamics model
    @DynamicsModelRegistry.register("custom_protocol_first")
    class CustomDynamicsModel:
        """
        A custom Protocol-Based dynamics model.

        This model directly implements the DynamicsModel Protocol without
        inheriting from any base class. It demonstrates how to create
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

        def steady_state(self, alpha, beta, gamma, **_):
            """
            Compute steady state values.

            Args:
                alpha: Transcription rate
                beta: Splicing rate
                gamma: Degradation rate
                **_: Additional parameters (ignored)

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
    """Explain the benefits of the Protocol-Based approach."""
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
    print("   - Protocol-Based implementation uses stateful objects with functional interfaces")
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

    print("=== Protocol-Based Architecture Example ===")
    print("This example demonstrates the Protocol-Based architecture in PyroVelocity's modular implementation.")

    # Demonstrate dynamics model
    demonstrate_dynamics_model()

    # Demonstrate standard model
    demonstrate_standard_model()

    # Create a custom Protocol-Based component
    create_custom_protocol_first_component()

    # Explain the benefits of the Protocol-Based approach
    explain_protocol_first_benefits()

    print("\n=== Example Completed ===")


if __name__ == "__main__":
    main()
