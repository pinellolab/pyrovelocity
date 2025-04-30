"""
Example script demonstrating the integration between PyroVelocityModel and different guide implementations.

This script shows how to:
1. Create a PyroVelocityModel with different guide implementations
2. Run inference with SVI and different guide types
3. Compare the results of different guides
4. Sample from the posterior and compute RNA velocity
"""

import torch
import pyro
import pyro.distributions as dist
import pyro.infer
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvelo as scv
from importlib.resources import files

# Import model creation and components
from pyrovelocity.models.modular.factory import create_model, standard_model_config
from pyrovelocity.models.modular.components.guides import AutoGuideFactory, NormalGuide, DeltaGuide
from pyrovelocity.models.modular.inference.unified import run_inference
from pyrovelocity.models.modular.inference.config import InferenceConfig
from pyrovelocity.models.modular.inference.posterior import analyze_posterior
from pyrovelocity.models.modular.registry import inference_guide_registry
from pyrovelocity.models.modular.model import PyroVelocityModel

# Import data loading utilities
from pyrovelocity.io.serialization import load_anndata_from_json

# Fixture hash for data validation
FIXTURE_HASH = "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae"

def load_test_data():
    """Load test data from the fixtures."""
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=str(fixture_file_path),
        expected_hash=FIXTURE_HASH,
    )

def main():
    # Set random seed for reproducibility
    pyro.set_rng_seed(0)
    torch.manual_seed(0)

    # 1. Load test data
    print("Loading test data...")
    adata = load_test_data()
    print(f"Loaded AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 2. Prepare data for velocity model
    print("Preparing data for velocity model...")
    # Set up AnnData using the direct method
    adata = PyroVelocityModel.setup_anndata(adata)

    # 3. Check available guide names in registry
    print("Available guides in registry:", inference_guide_registry.list_available())

    # 4. Create different guide configurations
    print("Creating different guide configurations...")
    # This uses the registered names in the registry: 'auto', 'normal', 'delta'
    guide_configs = {
        "Auto (AutoNormal)": {"name": "auto", "params": {"guide_type": "AutoNormal", "init_scale": 0.1}},
        "Auto (AutoDelta)": {"name": "auto", "params": {"guide_type": "AutoDelta", "init_scale": 0.1}},
        "Normal": {"name": "normal", "params": {"init_scale": 0.1}},
        "Delta": {"name": "delta", "params": {}},
    }

    # 5. Create model configurations with different guides
    print("Creating model configurations with different guides...")
    configs = {}
    for guide_name, guide_config in guide_configs.items():
        config = standard_model_config()
        config.inference_guide.name = guide_config["name"]
        config.inference_guide.params = guide_config["params"]
        configs[guide_name] = config

    # 6. Create models with different guides
    print("Creating models with different guides...")
    models = {}
    for guide_name, config in configs.items():
        print(f"Creating model with guide: {guide_name}")
        models[guide_name] = create_model(config)

    # 7. Train each model and store results
    print("Training models and storing results...")
    metrics = {}
    adatas = {}
    
    for name, model in models.items():
        print(f"Training model: {name}")
        # Train the model
        model.train(
            adata=adata,
            max_epochs=100,  # Reduced for example
            learning_rate=0.01,
            use_gpu=False,
        )
        
        # Generate posterior samples
        posterior_samples = model.generate_posterior_samples(
            adata=adata,
            num_samples=30
        )
        
        # Store results in AnnData
        adatas[name] = model.store_results_in_anndata(
            adata=adata.copy(),
            posterior_samples=posterior_samples
        )
        
        # Store metrics - note: we don't have direct access to training history yet
        # This is a placeholder for future implementation
        metrics[name] = {
            "adata": adatas[name],
            "elbo": None  # We'll need to implement a way to access training history
        }

    # 8. Print metrics
    print("\nInference Results Comparison:")
    print("-" * 80)
    print(f"{'Guide Name':<25} {'Final ELBO':<15}")
    print("-" * 80)
    for guide_name, result in metrics.items():
        elbo_val = result["elbo"] if result["elbo"] is not None else "N/A"
        print(f"{guide_name:<25} {elbo_val}")
    print("-" * 80)

    # 9. Select a model for demonstration (using the first one)
    demo_guide_name = list(models.keys())[0]
    demo_model = models[demo_guide_name]
    demo_adata = metrics[demo_guide_name]["adata"]
    
    print(f"\nUsing {demo_guide_name} for demonstration...")

    # 10. Create a figure showing latent time and velocity streams
    print("Creating visualization...")
    if 'velocity_model_latent_time' in demo_adata.obs.columns:
        sc.pl.umap(demo_adata, color="velocity_model_latent_time", title=f"Latent Time - {demo_guide_name}")

    # Check if 'clusters' exists in the AnnData object
    if 'clusters' in demo_adata.obs.columns:
        scv.pl.velocity_embedding_stream(demo_adata, basis="umap", color="clusters", title=f"Velocity - {demo_guide_name}")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(demo_adata, basis="umap", title=f"Velocity - {demo_guide_name}")

    # 11. Demonstrate a simple stand-alone Pyro model and guide
    print("\nDemonstrating a simple stand-alone Pyro model and guide...")
    
    # Section header
    print("\n" + "="*80)
    print("SIMPLE PYRO MODEL DEMONSTRATION")
    print("="*80)
    
    # Clear pyro param store to avoid conflicts
    pyro.clear_param_store()
    
    # Define a very simple model (just one parameter)
    def simple_model(data=None):
        # Sample from prior
        mu = pyro.sample("mu", dist.Normal(0.0, 10.0))
        return mu
    
    # Create a guide for our model using AutoNormal
    auto_normal = pyro.infer.autoguide.AutoNormal(simple_model)
    
    # Run the guide to see the initial distribution
    print("Initial guide distribution:")
    guide_trace = pyro.poutine.trace(auto_normal).get_trace()
    print(guide_trace.nodes["mu"]["fn"])
    
    # Create a small toy dataset - we'll use this for "training"
    # We'll pretend we observed values with mean 3.0
    observed_data = torch.tensor([3.0])
    
    # Define a model that includes observations
    def model_with_obs(data=None):
        # Sample from prior
        mu = pyro.sample("mu", dist.Normal(0.0, 10.0))
        # Add observation
        if data is not None:
            pyro.sample("obs", dist.Normal(mu, 0.1), obs=data)
        return mu
    
    # Create a guide
    guide = pyro.infer.autoguide.AutoNormal(model_with_obs)
    
    # Training the guide with SVI
    optimizer = pyro.optim.Adam({"lr": 0.01})
    svi = pyro.infer.SVI(model_with_obs, guide, optimizer, loss=pyro.infer.Trace_ELBO())
    
    losses = []
    num_steps = 100
    for step in range(num_steps):
        loss = svi.step(observed_data)
        losses.append(loss)
        if step % 20 == 0:
            print(f"Step {step}: loss = {loss:.4f}")
    
    # Display the trained guide
    print("\nTrained guide distribution:")
    for name, param in guide.named_parameters():
        print(f"{name}: {param.data}")
    
    # Correct parameter name here - inspect the param_store to get actual names
    pyro_params = dict(pyro.get_param_store().items())
    print("\nAll parameters in param_store:")
    for name in pyro_params:
        print(name)
    
    # Access the parameter using the correct name from param_store
    try:
        # Use the first parameter name that contains 'loc'
        loc_param_name = next(name for name in pyro_params if 'loc' in name)
        loc = pyro.param(loc_param_name)
        scale_param_name = next(name for name in pyro_params if 'scale' in name)
        scale = pyro.param(scale_param_name)
        
        print(f"\nPosterior mean: {loc.item():.4f}")
        print(f"Posterior scale: {scale.item():.4f}")
        print(f"True mean: {observed_data.item():.4f}")
        
        # Sample from the posterior
        predictive = pyro.infer.Predictive(model_with_obs, guide=guide, num_samples=1000)
        posterior_samples = predictive(observed_data)
        
        # Visualize the posterior samples
        plt.figure(figsize=(10, 6))
        plt.hist(posterior_samples["mu"].numpy(), bins=30, alpha=0.7, label="Posterior samples")
        plt.axvline(observed_data.item(), color="r", linestyle="--", label=f"True value: {observed_data.item():.4f}")
        plt.axvline(loc.item(), color="g", linestyle="-", label=f"Posterior mean: {loc.item():.4f}")
        plt.legend()
        plt.title("Posterior Distribution")
        plt.xlabel("Parameter Value")
        plt.ylabel("Frequency")
        plt.savefig("posterior_samples.png")
        print("\nPosterior samples plot saved to posterior_samples.png")
        
        # Plot the training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title("SVI Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig("svi_loss.png")
        print("Training loss plot saved to svi_loss.png")
    
    except (StopIteration, KeyError) as e:
        print(f"\nError accessing parameters: {e}")
        print("Could not generate posterior plots due to parameter access error")

    print("\nModel-guide integration example completed!")

if __name__ == "__main__":
    main() 
