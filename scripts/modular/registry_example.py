#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating the use of the registry system in the PyroVelocity modular architecture.

This script shows how to:
1. Register custom components in the registry
2. List available components
3. Create components from the registry
4. Use the components in a model
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

from typing import Any, Dict, Optional, Tuple


# Define a custom dynamics model
@DynamicsModelRegistry.register("custom_linear")
class CustomLinearDynamicsModel(DynamicsModel):
    """
    A custom linear dynamics model that implements the DynamicsModel interface.

    This model uses a simple linear relationship between unspliced and spliced RNA.
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


# Define a custom prior model
@PriorModelRegistry.register("custom_normal")
class CustomNormalPriorModel(PriorModel):
    """
    A custom prior model that uses normal distributions for all parameters.
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

    # Create a dynamics model
    dynamics_model = DynamicsModelRegistry.create("custom_linear")
    print(f"Created dynamics model: {dynamics_model.__class__.__name__}")

    # Create a prior model
    prior_model = PriorModelRegistry.create("custom_normal")
    print(f"Created prior model: {prior_model.__class__.__name__}")

    # Create a likelihood model
    likelihood_model = LikelihoodModelRegistry.create("poisson")
    print(f"Created likelihood model: {likelihood_model.__class__.__name__}")

    # Create an observation model
    observation_model = ObservationModelRegistry.create("standard")
    print(f"Created observation model: {observation_model.__class__.__name__}")

    # Create an inference guide
    inference_guide = InferenceGuideRegistry.create("auto")
    print(f"Created inference guide: {inference_guide.__class__.__name__}")

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

    print("\nRegistry example completed successfully!")


if __name__ == "__main__":
    main()
