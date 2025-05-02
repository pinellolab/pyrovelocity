"""
Example script demonstrating the use of the unified configuration system in PyroVelocity.

This script shows how to:
1. Create components using the ComponentFactory
2. Create models using the unified configuration system
3. Customize model configurations
4. Use the builder pattern for fluent configuration
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData

from pyrovelocity.models.modular.config import (
    ComponentConfig,
    ModelConfig,
    ComponentType,
)
from pyrovelocity.models.modular.factory import (
    ComponentFactory,
    create_model_from_config,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


def create_test_data(n_cells=50, n_genes=10):
    """Create synthetic test data for demonstration."""
    # Create random data
    u_data = np.random.poisson(10, size=(n_cells, n_genes))
    s_data = np.random.poisson(20, size=(n_cells, n_genes))

    # Create AnnData object
    adata = AnnData(
        X=s_data,
        layers={
            "unspliced": u_data,
            "spliced": s_data,
        },
    )

    # Add gene names
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Add UMAP coordinates for visualization
    adata.obsm = {}
    adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_cells, 2))

    print(f"Created synthetic AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")
    return adata


def example_1_standard_model():
    """Example 1: Creating a standard model using the unified configuration system."""
    print("\n=== Example 1: Standard Model ===")

    # Create a standard model configuration
    config = ModelConfig.standard()
    print("Created standard model configuration:")
    print(f"  Dynamics model: {config.dynamics_model.name}")
    print(f"  Prior model: {config.prior_model.name}")
    print(f"  Likelihood model: {config.likelihood_model.name}")
    print(f"  Observation model: {config.observation_model.name}")
    print(f"  Inference guide: {config.inference_guide.name}")

    # Create a model from the configuration
    model = create_model_from_config(config)
    print("Created model from configuration")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Dynamics model type: {type(model.dynamics_model).__name__}")
    print(f"  Prior model type: {type(model.prior_model).__name__}")
    print(f"  Likelihood model type: {type(model.likelihood_model).__name__}")
    print(f"  Observation model type: {type(model.observation_model).__name__}")
    print(f"  Guide model type: {type(model.guide_model).__name__}")

    return model


def example_2_custom_model():
    """Example 2: Creating a custom model using the unified configuration system."""
    print("\n=== Example 2: Custom Model ===")

    # Create custom component configurations
    dynamics_config = ComponentConfig(
        name="standard",
        params={"use_analytical_solution": True},
    )
    prior_config = ComponentConfig(
        name="lognormal",
        params={"scale_alpha": 1.0, "scale_beta": 0.5, "scale_gamma": 1.5},
    )
    likelihood_config = ComponentConfig(
        name="poisson",
        params={},  # PoissonLikelihoodModel doesn't have a use_zero_inflation parameter
    )
    observation_config = ComponentConfig(
        name="standard",
        params={},  # StandardObservationModel doesn't have a normalize parameter
    )
    inference_config = ComponentConfig(
        name="auto",
        params={},  # AutoGuideFactory doesn't have an init_loc_fn parameter
    )

    # Create a model configuration
    config = ModelConfig(
        dynamics_model=dynamics_config,
        prior_model=prior_config,
        likelihood_model=likelihood_config,
        observation_model=observation_config,
        inference_guide=inference_config,
        metadata={"description": "Custom model with specific parameters"},
    )

    print("Created custom model configuration:")
    print(f"  Dynamics model: {config.dynamics_model.name}")
    print(f"  Prior model: {config.prior_model.name}")
    print(f"  Likelihood model: {config.likelihood_model.name}")
    print(f"  Observation model: {config.observation_model.name}")
    print(f"  Inference guide: {config.inference_guide.name}")
    print(f"  Metadata: {config.metadata}")

    # Create a model from the configuration
    model = create_model_from_config(config)
    print("Created model from configuration")
    print(f"  Model type: {type(model).__name__}")

    return model


def example_3_component_factory():
    """Example 3: Using the ComponentFactory directly."""
    print("\n=== Example 3: Component Factory ===")

    # Create a dynamics model using the ComponentFactory
    dynamics_config = ComponentConfig(
        name="standard",
        params={"use_analytical_solution": True},
    )
    dynamics_model = ComponentFactory.create_dynamics_model(dynamics_config)
    print(f"Created dynamics model: {type(dynamics_model).__name__}")

    # Create a prior model using the ComponentFactory
    prior_config = ComponentConfig(
        name="lognormal",
        params={"scale_alpha": 1.0, "scale_beta": 0.5, "scale_gamma": 1.5},
    )
    prior_model = ComponentFactory.create_prior_model(prior_config)
    print(f"Created prior model: {type(prior_model).__name__}")

    # Create a likelihood model using the ComponentFactory
    likelihood_config = ComponentConfig(
        name="poisson",
        params={},  # PoissonLikelihoodModel doesn't have a use_zero_inflation parameter
    )
    likelihood_model = ComponentFactory.create_likelihood_model(likelihood_config)
    print(f"Created likelihood model: {type(likelihood_model).__name__}")

    # Create an observation model using the ComponentFactory
    observation_config = ComponentConfig(
        name="standard",
        params={},  # StandardObservationModel doesn't have a normalize parameter
    )
    observation_model = ComponentFactory.create_observation_model(observation_config)
    print(f"Created observation model: {type(observation_model).__name__}")

    # Create an inference guide using the ComponentFactory
    inference_config = ComponentConfig(
        name="auto",
        params={},  # AutoGuideFactory doesn't have an init_loc_fn parameter
    )
    inference_guide = ComponentFactory.create_inference_guide(inference_config)
    print(f"Created inference guide: {type(inference_guide).__name__}")

    # Create a model manually
    model = PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        observation_model=observation_model,
        guide_model=inference_guide,
    )
    print(f"Created model manually: {type(model).__name__}")

    return model


def example_4_dictionary_config():
    """Example 4: Creating a model from a dictionary configuration."""
    print("\n=== Example 4: Dictionary Configuration ===")

    # Create a configuration dictionary
    config_dict = {
        "dynamics_model": {"name": "standard", "params": {"use_analytical_solution": True}},
        "prior_model": {"name": "lognormal", "params": {"scale_alpha": 1.0, "scale_beta": 0.5, "scale_gamma": 1.5}},
        "likelihood_model": {"name": "poisson", "params": {}},
        "observation_model": {"name": "standard", "params": {}},
        "inference_guide": {"name": "auto", "params": {}},
        "metadata": {"description": "Model created from dictionary configuration"},
    }

    print("Created dictionary configuration:")
    print(f"  Dynamics model: {config_dict['dynamics_model']['name']}")
    print(f"  Prior model: {config_dict['prior_model']['name']}")
    print(f"  Likelihood model: {config_dict['likelihood_model']['name']}")
    print(f"  Observation model: {config_dict['observation_model']['name']}")
    print(f"  Inference guide: {config_dict['inference_guide']['name']}")
    print(f"  Metadata: {config_dict['metadata']}")

    # Create a model from the dictionary configuration
    model = create_model_from_config(config_dict)
    print("Created model from dictionary configuration")
    print(f"  Model type: {type(model).__name__}")

    return model


def example_5_train_model():
    """Example 5: Training a model created with the unified configuration system."""
    print("\n=== Example 5: Training a Model ===")

    # Set random seed for reproducibility
    pyro.set_rng_seed(0)
    torch.manual_seed(0)

    # Create test data
    adata = create_test_data(n_cells=50, n_genes=10)

    # Create a standard model
    model = create_model_from_config(ModelConfig.standard())
    print(f"Created model: {type(model).__name__}")

    # Prepare AnnData for PyroVelocity
    adata = PyroVelocityModel.setup_anndata(adata)
    print("Prepared AnnData for PyroVelocity")

    # Train the model with minimal epochs for demonstration
    print("Training model...")
    model.train(
        adata=adata,
        max_epochs=5,  # Use small number for demonstration
        batch_size=10,
        learning_rate=0.01,
        use_gpu=False,  # Set to True if GPU is available
    )
    print("Model training complete")

    # Generate posterior samples
    print("Generating posterior samples...")
    posterior_samples = model.generate_posterior_samples(
        adata=adata,
        num_samples=10,  # Use small number for demonstration
    )
    print("Generated posterior samples")

    # Store results in AnnData
    print("Storing results in AnnData...")
    adata = model.store_results_in_anndata(
        adata=adata,
        posterior_samples=posterior_samples,
        model_name="velocity_model",  # Prefix for stored results
    )
    print("Results stored in AnnData")

    return adata, model


def main():
    """Run all examples."""
    print("=== PyroVelocity Unified Configuration System Examples ===")

    # Example 1: Standard model
    model1 = example_1_standard_model()

    # Example 2: Custom model
    model2 = example_2_custom_model()

    # Example 3: Component factory
    model3 = example_3_component_factory()

    # Example 4: Dictionary configuration
    model4 = example_4_dictionary_config()

    # Example 5: Train model
    adata, model5 = example_5_train_model()

    print("\n=== All Examples Complete ===")


if __name__ == "__main__":
    main()
