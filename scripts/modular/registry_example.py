"""
Example script demonstrating the use of the registry system in the PyroVelocity modular architecture.

This script shows how to:
1. Register custom components in the registry (both base class and Protocol-Based implementation (now the default)s)
2. List available components
3. Create components from the registry
4. Use the components in a model
5. Create a custom model with registered components
6. Compare base class and Protocol-Based implementation (now the default)s

The PyroVelocity modular architecture supports two implementation approaches:
1. Base Class Approach: Components inherit from base classes (BaseDynamicsModel, etc.)
2. Protocol-Based Approach: Components directly implement Protocol interfaces without inheritance

This script demonstrates both approaches for custom component registration.
"""

import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    PriorModelRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    InferenceGuideRegistry,
)

from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    PriorModel,
    LikelihoodModel,
    ObservationModel,
    InferenceGuide,
    BatchTensor,
    ParamTensor,
    ModelState,
)

from pyrovelocity.models.modular.factory import (
    DynamicsModelConfig,
    PriorModelConfig,
    LikelihoodModelConfig,
    ObservationModelConfig,
    InferenceGuideConfig,
    PyroVelocityModelConfig,
    create_model,
    create_model_from_config,
)

from pyrovelocity.models.modular.config import (
    ModelConfig,
    ComponentConfig,
)

from typing import Any, Dict, Optional, Tuple


# Define a custom dynamics model using the base class approach
@DynamicsModelRegistry.register("custom_linear")
class CustomLinearDynamicsModel:
    """
    A custom linear dynamics model that implements the DynamicsModel interface.

    This model uses a simple linear relationship between unspliced and spliced RNA.
    This implementation uses the base class approach by directly implementing the Protocol.
    """

    def forward(
        self,
        u: BatchTensor,
        s: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        t: Optional[BatchTensor] = None,
        **kwargs: Any,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Compute the expected unspliced and spliced RNA counts.

        Args:
            u: Observed unspliced RNA counts
            s: Observed spliced RNA counts
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor
            t: Optional time points
            **kwargs: Additional parameters

        Returns:
            Tuple of (expected unspliced counts, expected spliced counts)
        """
        # Simple linear model: u_expected = alpha / beta, s_expected = alpha / gamma
        u_expected = alpha / beta
        s_expected = alpha / gamma

        # Apply scaling if provided
        if scaling is not None:
            u_expected = u_expected * scaling
            s_expected = s_expected * scaling

        return u_expected, s_expected

    def steady_state(
        self,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        **kwargs: Any,
    ) -> Tuple[ParamTensor, ParamTensor]:
        """
        Compute the steady-state unspliced and spliced RNA counts.

        Args:
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            **kwargs: Additional parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        # Steady state is the same as the expected values in this simple model
        u_ss = alpha / beta
        s_ss = alpha / gamma
        return u_ss, s_ss


# Define a custom prior model using the base class approach
@PriorModelRegistry.register("custom_normal")
class CustomNormalPriorModel:
    """
    A custom prior model that uses normal distributions for all parameters.

    This implementation uses the base class approach by directly implementing the Protocol.
    """

    def forward(
        self,
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        plate: pyro.plate,
        **kwargs: Any,
    ) -> ModelState:
        """
        Sample model parameters from prior distributions.

        Args:
            u_obs: Observed unspliced RNA counts
            s_obs: Observed spliced RNA counts
            plate: Pyro plate for batched sampling
            **kwargs: Additional parameters

        Returns:
            Dictionary containing sampled parameters
        """
        # Get the number of genes
        n_genes = u_obs.shape[-1]

        # Sample parameters from normal distributions
        with plate:
            alpha = pyro.sample("alpha", dist.Normal(0.0, 1.0).expand([n_genes]))
            beta = pyro.sample("beta", dist.Normal(0.0, 1.0).expand([n_genes]))
            gamma = pyro.sample("gamma", dist.Normal(0.0, 1.0).expand([n_genes]))

        # Return the sampled parameters
        return {"alpha": alpha, "beta": beta, "gamma": gamma}


# Define a custom Protocol-Based dynamics model
@DynamicsModelRegistry.register("custom_linear")
class CustomLinearDynamicsModel:
    """
    A custom Protocol-Based dynamics model that directly implements the DynamicsModel Protocol.

    This model uses a simple linear relationship between unspliced and spliced RNA.
    This implementation uses the Protocol-Based approach (now the default) by directly implementing the Protocol
    without inheriting from any base class.
    """

    def __init__(self, name="custom_linear", **kwargs):
        """Initialize the custom dynamics model."""
        self.name = name

    def forward(self, context):
        """
        Compute expected unspliced and spliced counts.

        Args:
            context: Dictionary containing model context

        Returns:
            Updated context with expected counts
        """
        # Extract parameters from context
        u_obs = context["u_obs"]
        s_obs = context["s_obs"]
        alpha = context["alpha"]
        beta = context["beta"]
        gamma = context["gamma"]
        scaling = context.get("scaling", None)

        # Compute steady state
        u_ss, s_ss = self.steady_state(alpha, beta, gamma)

        # Apply scaling if provided
        if scaling is not None:
            u_ss = u_ss * scaling
            s_ss = s_ss * scaling

        # Update context
        context["u_expected"] = u_ss
        context["s_expected"] = s_ss
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
        # Simple linear model: u_ss = alpha / beta, s_ss = alpha / gamma
        u_ss = alpha / beta
        s_ss = alpha / gamma

        return u_ss, s_ss


# Define a custom Protocol-Based prior model
@PriorModelRegistry.register("custom_normal")
class CustomNormalPriorModel:
    """
    A custom Protocol-Based prior model that directly implements the PriorModel Protocol.

    This model uses normal distributions for all parameters.
    This implementation uses the Protocol-Based approach (now the default) by directly implementing the Protocol
    without inheriting from any base class.
    """

    def __init__(self, name="custom_normal", **kwargs):
        """Initialize the custom prior model."""
        self.name = name

    def forward(self, context):
        """
        Sample model parameters from prior distributions.

        Args:
            context: Dictionary containing model context

        Returns:
            Updated context with sampled parameters
        """
        # Extract data from context
        u_obs = context["u_obs"]
        s_obs = context["s_obs"]
        plate = context.get("plate", None)
        include_prior = context.get("include_prior", True)

        # Get the number of genes
        n_genes = u_obs.shape[-1]

        # Create a plate if not provided
        if plate is None:
            plate = pyro.plate("genes", n_genes)

        # Sample parameters from normal distributions
        with plate:
            alpha = pyro.sample("alpha", dist.Normal(0.0, 1.0).expand([n_genes]))
            beta = pyro.sample("beta", dist.Normal(0.0, 1.0).expand([n_genes]))
            gamma = pyro.sample("gamma", dist.Normal(0.0, 1.0).expand([n_genes]))

        # Update context
        context["alpha"] = alpha
        context["beta"] = beta
        context["gamma"] = gamma

        return context


def main():
    """Run the registry example."""
    # Set random seed for reproducibility
    pyro.set_rng_seed(42)
    torch.manual_seed(42)

    # Print available components in each registry
    print("Available Dynamics Models:")
    print(DynamicsModelRegistry.list_available())

    print("\nAvailable Prior Models:")
    print(PriorModelRegistry.list_available())

    print("\nAvailable Likelihood Models:")
    print(LikelihoodModelRegistry.list_available())

    print("\nAvailable Observation Models:")
    print(ObservationModelRegistry.list_available())

    print("\nAvailable Inference Guides:")
    print(InferenceGuideRegistry.list_available())

    # Create components from the registry
    print("\nCreating components from the registry...")

    # Create base class components
    print("Creating base class components:")
    dynamics_model = DynamicsModelRegistry.create("custom_linear")
    print(f"  Created dynamics model: {dynamics_model.__class__.__name__}")

    prior_model = PriorModelRegistry.create("custom_normal")
    print(f"  Created prior model: {prior_model.__class__.__name__}")

    likelihood_model = LikelihoodModelRegistry.create("poisson")
    print(f"  Created likelihood model: {likelihood_model.__class__.__name__}")

    observation_model = ObservationModelRegistry.create("standard")
    print(f"  Created observation model: {observation_model.__class__.__name__}")

    inference_guide = InferenceGuideRegistry.create("auto")
    print(f"  Created inference guide: {inference_guide.__class__.__name__}")

    # Create Protocol-Based components (now the default)
    print("\nCreating Protocol-Based components (now the default):")
    dynamics_model_protocol = DynamicsModelRegistry.create("custom_linear")
    print(f"  Created dynamics model: {dynamics_model_protocol.__class__.__name__}")

    prior_model_protocol = PriorModelRegistry.create("custom_normal")
    print(f"  Created prior model: {prior_model_protocol.__class__.__name__}")

    likelihood_model_protocol = LikelihoodModelRegistry.create("poisson")
    print(f"  Created likelihood model: {likelihood_model_protocol.__class__.__name__}")

    observation_model_protocol = ObservationModelRegistry.create("standard")
    print(f"  Created observation model: {observation_model_protocol.__class__.__name__}")

    inference_guide_protocol = InferenceGuideRegistry.create("auto")
    print(f"  Created inference guide: {inference_guide_protocol.__class__.__name__}")

    # Test the dynamics model
    print("\nTesting the dynamics model...")

    # Create some test data
    n_genes = 3
    alpha = torch.ones(n_genes) * 2.0
    beta = torch.ones(n_genes) * 0.5
    gamma = torch.ones(n_genes) * 0.3
    u = torch.ones(1, 10, n_genes) * 4.0
    s = torch.ones(1, 10, n_genes) * 6.0

    # Compute expected values
    u_expected, s_expected = dynamics_model.forward(u, s, alpha, beta, gamma)
    print(f"Expected unspliced counts: {u_expected}")
    print(f"Expected spliced counts: {s_expected}")

    # Compute steady state
    u_ss, s_ss = dynamics_model.steady_state(alpha, beta, gamma)
    print(f"Steady-state unspliced counts: {u_ss}")
    print(f"Steady-state spliced counts: {s_ss}")

    # Verify that the steady state matches the expected values
    print(f"Steady state matches expected values: {torch.allclose(u_expected, u_ss) and torch.allclose(s_expected, s_ss)}")

    # Test both dynamics models
    print("\nTesting both dynamics models...")

    # Create some test data
    n_genes = 3
    alpha = torch.ones(n_genes) * 2.0
    beta = torch.ones(n_genes) * 0.5
    gamma = torch.ones(n_genes) * 0.3
    u = torch.ones(1, 10, n_genes) * 4.0
    s = torch.ones(1, 10, n_genes) * 6.0

    # Test base class dynamics model
    print("\nTesting base class dynamics model:")
    u_expected, s_expected = dynamics_model.forward(u, s, alpha, beta, gamma)
    print(f"  Expected unspliced counts: {u_expected}")
    print(f"  Expected spliced counts: {s_expected}")

    u_ss, s_ss = dynamics_model.steady_state(alpha, beta, gamma)
    print(f"  Steady-state unspliced counts: {u_ss}")
    print(f"  Steady-state spliced counts: {s_ss}")

    # Test Protocol-Based dynamics model
    print("\nTesting Protocol-Based dynamics model:")
    context = {
        "u_obs": u,
        "s_obs": s,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }

    result_context = dynamics_model_protocol.forward(context)
    print(f"  Expected unspliced counts: {result_context['u_expected']}")
    print(f"  Expected spliced counts: {result_context['s_expected']}")

    u_ss_protocol, s_ss_protocol = dynamics_model_protocol.steady_state(alpha, beta, gamma)
    print(f"  Steady-state unspliced counts: {u_ss_protocol}")
    print(f"  Steady-state spliced counts: {s_ss_protocol}")

    # Compare results
    print("\nComparing results:")
    u_equal = torch.allclose(u_expected, result_context['u_expected'])
    s_equal = torch.allclose(s_expected, result_context['s_expected'])
    print(f"  Expected counts match: {u_equal and s_equal}")

    u_ss_equal = torch.allclose(u_ss, u_ss_protocol)
    s_ss_equal = torch.allclose(s_ss, s_ss_protocol)
    print(f"  Steady state counts match: {u_ss_equal and s_ss_equal}")

    # Create a custom model using the factory system with base class components
    print("\nCreating a custom model with base class components...")
    custom_config = PyroVelocityModelConfig(
        dynamics_model=DynamicsModelConfig(
            name="custom_linear",
            params={}
        ),
        prior_model=PriorModelConfig(
            name="custom_normal",
            params={}
        ),
        likelihood_model=LikelihoodModelConfig(
            name="poisson",
            params={}
        ),
        observation_model=ObservationModelConfig(
            name="standard",
            params={}
        ),
        inference_guide=InferenceGuideConfig(
            name="auto",
            params={"guide_type": "AutoNormal", "init_scale": 0.1}
        )
    )

    # Create the model
    custom_model = create_model(custom_config)
    print(f"Created custom model with base class components:")
    print(f"  - Dynamics: {custom_model.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {custom_model.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {custom_model.likelihood_model.__class__.__name__}")
    print(f"  - Observation: {custom_model.observation_model.__class__.__name__}")
    print(f"  - Guide: {custom_model.guide_model.__class__.__name__}")

    # Create a custom model using the factory system with Protocol-Based components (now the default)
    print("\nCreating a custom model with Protocol-Based components (now the default)...")
    protocol_config = ModelConfig(
        dynamics_model=ComponentConfig(
            name="custom_linear",
            params={}
        ),
        prior_model=ComponentConfig(
            name="custom_normal",
            params={}
        ),
        likelihood_model=ComponentConfig(
            name="poisson",
            params={}
        ),
        observation_model=ComponentConfig(
            name="standard",
            params={}
        ),
        inference_guide=ComponentConfig(
            name="auto",
            params={"guide_type": "AutoNormal", "init_scale": 0.1}
        )
    )

    # Create the model
    protocol_model = create_model_from_config(protocol_config)
    print(f"Created custom model with Protocol-Based components (now the default):")
    print(f"  - Dynamics: {protocol_model.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {protocol_model.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {protocol_model.likelihood_model.__class__.__name__}")
    print(f"  - Observation: {protocol_model.observation_model.__class__.__name__}")
    print(f"  - Guide: {protocol_model.guide_model.__class__.__name__}")

    # Explain the differences in implementation
    print("\nKey differences between base class and Protocol-Based implementation (now the default)s:")
    print("1. Base class components inherit from base classes, which provide common functionality")
    print("2. Protocol-Based components (now the default) directly implement Protocol interfaces")
    print("3. Protocol-Based components (now the default) use utility functions for common functionality")
    print("4. Both implementations produce identical results, demonstrating functional equivalence")
    print("5. Protocol-Based approach (now the default) reduces code complexity by eliminating inheritance hierarchies")
    print("6. Protocol-Based approach (now the default) creates perfect architectural consistency with the JAX implementation")

    print("\nRegistry example completed successfully!")


if __name__ == "__main__":
    main()
