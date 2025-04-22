"""
Example of RNA velocity analysis using PyroVelocity PyTorch/Pyro modular factory system.

This example demonstrates:
1. Creating a model using the factory system
2. Running inference with the model
3. Analyzing and visualizing results
"""

import torch
import pyro

from pyrovelocity.models.modular.factory import (
    # Factory system
    DynamicsModelConfig,
    PriorModelConfig,
    LikelihoodModelConfig,
    ObservationModelConfig,
    InferenceGuideConfig,
    PyroVelocityModelConfig,
    create_model,
    create_standard_model,
    standard_model_config,
)


def main():
    # Set random seed for reproducibility
    pyro.set_rng_seed(0)
    torch.manual_seed(0)

    # Generate synthetic data directly
    n_cells = 100
    n_genes = 10
    batch_size = 1

    # Create random data
    u_obs = torch.poisson(torch.ones((batch_size, n_cells, n_genes)) * 5.0)
    s_obs = torch.poisson(torch.ones((batch_size, n_cells, n_genes)) * 5.0)

    # Print shapes and values
    print(f"u_obs shape: {u_obs.shape}, min: {torch.min(u_obs)}, max: {torch.max(u_obs)}")
    print(f"s_obs shape: {s_obs.shape}, min: {torch.min(s_obs)}, max: {torch.max(s_obs)}")

    # Method 1: Create a standard model using the factory system
    print("Method 1: Using create_standard_model()")
    model1 = create_standard_model()
    print(f"Model 1: {model1}")

    # Method 2: Create a model using the standard configuration
    print("Method 2: Using standard_model_config() and create_model()")
    config = standard_model_config()
    model2 = create_model(config)
    print(f"Model 2: {model2}")

    # Method 3: Create a custom model using the factory system
    print("Method 3: Using custom configuration and create_model()")
    custom_config = PyroVelocityModelConfig(
        dynamics_model=DynamicsModelConfig(
            name="standard",
            params={"shared_time": True}
        ),
        prior_model=PriorModelConfig(
            name="lognormal",
            params={"scale_alpha": 2.0, "scale_beta": 0.5, "scale_gamma": 1.5}
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
            params={}
        )
    )
    model3 = create_model(custom_config)
    print(f"Model 3: {model3}")

    # Print model information
    print("\nModel Information:")
    print(f"Model 3 dynamics model: {model3.dynamics_model.__class__.__name__}")
    print(f"Model 3 prior model: {model3.prior_model.__class__.__name__}")
    print(f"Model 3 likelihood model: {model3.likelihood_model.__class__.__name__}")
    print(f"Model 3 observation model: {model3.observation_model.__class__.__name__}")
    print(f"Model 3 guide model: {model3.guide_model.__class__.__name__}")

    # Get the model state
    state = model3.get_state()
    print("\nModel state:")
    print(f"  dynamics_state: {type(state.dynamics_state)}")
    print(f"  prior_state: {type(state.prior_state)}")
    print(f"  likelihood_state: {type(state.likelihood_state)}")
    print(f"  observation_state: {type(state.observation_state)}")
    print(f"  guide_state: {type(state.guide_state)}")
    print(f"  metadata: {state.metadata}")

    print("Done!")

if __name__ == "__main__":
    main()
