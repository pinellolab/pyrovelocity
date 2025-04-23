"""
Example of RNA velocity analysis using PyroVelocity PyTorch/Pyro modular factory system.

This example demonstrates:
1. Creating models using different factory methods
2. Customizing model configurations
3. Inspecting model components and state
4. Running a simple forward pass
"""

import torch
import pyro
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    PriorModelRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    InferenceGuideRegistry,
)

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

    # List available components in each registry
    print("Available components in registries:")
    print(f"Dynamics Models: {DynamicsModelRegistry.list_available()}")
    print(f"Prior Models: {PriorModelRegistry.list_available()}")
    print(f"Likelihood Models: {LikelihoodModelRegistry.list_available()}")
    print(f"Observation Models: {ObservationModelRegistry.list_available()}")
    print(f"Inference Guides: {InferenceGuideRegistry.list_available()}")
    print()

    # Generate synthetic data directly
    n_cells = 100
    n_genes = 10
    batch_size = 1

    # Create random data
    u_obs = torch.poisson(torch.ones((batch_size, n_cells, n_genes)) * 5.0)
    s_obs = torch.poisson(torch.ones((batch_size, n_cells, n_genes)) * 5.0)

    # Print shapes and values
    print(f"u_obs shape: {u_obs.shape}, min: {torch.min(u_obs).item():.2f}, max: {torch.max(u_obs).item():.2f}")
    print(f"s_obs shape: {s_obs.shape}, min: {torch.min(s_obs).item():.2f}, max: {torch.max(s_obs).item():.2f}")
    print()

    # Method 1: Create a standard model using the factory system
    print("Method 1: Using create_standard_model()")
    model1 = create_standard_model()
    print(f"Model 1: {model1}")
    print(f"  - Dynamics: {model1.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {model1.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {model1.likelihood_model.__class__.__name__}")
    print(f"  - Observation: {model1.observation_model.__class__.__name__}")
    print(f"  - Guide: {model1.guide_model.__class__.__name__}")
    print()

    # Method 2: Create a model using the standard configuration
    print("Method 2: Using standard_model_config() and create_model()")
    config = standard_model_config()
    print("Standard model configuration:")
    print(f"  - Dynamics: {config.dynamics_model.name}")
    print(f"  - Prior: {config.prior_model.name}")
    print(f"  - Likelihood: {config.likelihood_model.name}")
    print(f"  - Observation: {config.observation_model.name}")
    print(f"  - Guide: {config.inference_guide.name}")
    model2 = create_model(config)
    print(f"Model 2: {model2}")
    print()

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
            params={"guide_type": "AutoNormal", "init_scale": 0.1}
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
    
    # Run a simple forward pass with the dynamics model directly
    print("\nRunning a simple forward pass with the dynamics model:")
    try:
        # Sample some parameters
        n_genes = u_obs.shape[-1]
        alpha = torch.ones(n_genes) * 2.0
        beta = torch.ones(n_genes) * 0.5
        gamma = torch.ones(n_genes) * 0.3
        
        # Create a context dictionary
        context = {}
        context["u_obs"] = u_obs
        context["s_obs"] = s_obs
        
        # Run the dynamics model forward
        with torch.no_grad():
            # First prepare the context with the observation model
            context = model3.observation_model.forward(**context)
            
            # Then run the dynamics model
            u_expected, s_expected = model3.dynamics_model.forward(
                alpha=alpha, 
                beta=beta, 
                gamma=gamma,
                **context
            )
            
        print("Forward pass successful!")
        print(f"u_expected shape: {u_expected.shape}, min: {torch.min(u_expected).item():.2f}, max: {torch.max(u_expected).item():.2f}")
        print(f"s_expected shape: {s_expected.shape}, min: {torch.min(s_expected).item():.2f}, max: {torch.max(s_expected).item():.2f}")
    
    except Exception as e:
        print(f"Error during forward pass: {e}")

    print("\nDone!")

if __name__ == "__main__":
    main()
