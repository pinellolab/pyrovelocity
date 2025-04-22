"""
Example of model comparison and selection using PyroVelocity JAX/NumPyro implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating multiple velocity models with different configurations
3. Running inference on each model
4. Comparing models using information criteria
5. Selecting the best model
6. Analyzing and visualizing results
"""

import jax
import jax.numpy as jnp
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files

# Import core components
from pyrovelocity.models.jax.core.utils import create_key
from pyrovelocity.models.jax.core.state import InferenceConfig

# Import model creation
from pyrovelocity.models.jax.factory.factory import create_model
from pyrovelocity.models.jax.factory.config import (
    ModelConfig,
    DynamicsFunctionConfig,
    PriorFunctionConfig,
    LikelihoodFunctionConfig,
    ObservationFunctionConfig,
    GuideFunctionConfig,
)

# Import data processing
from pyrovelocity.models.jax.data.anndata import prepare_anndata

# Import inference
from pyrovelocity.models.jax.inference.unified import run_inference

# Import model comparison and selection
from pyrovelocity.models.jax.comparison import (
    compare_models,
    select_best_model,
)

# Import analysis
from pyrovelocity.models.jax.inference.posterior import compute_velocity, analyze_posterior

# Import visualization
from pyrovelocity.models.jax.data.anndata import store_results as format_anndata_output

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
    key = create_key(0)

    # 1. Load test data
    print("Loading test data...")
    adata = load_test_data()
    print(f"Loaded AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 2. Prepare data for velocity model
    print("Preparing data for velocity model...")
    data_dict = prepare_anndata(
        adata,
        spliced_layer="spliced",
        unspliced_layer="unspliced",
    )

    # 3. Create multiple model configurations
    print("Creating model configurations...")

    # Model 1: Standard model with lognormal priors and Poisson likelihood
    model_config1 = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )

    # Model 2: Standard model with informative priors and Poisson likelihood
    model_config2 = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="informative"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )

    # Model 3: Standard model with lognormal priors and negative binomial likelihood
    model_config3 = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="negative_binomial"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )

    # 4. Create inference configuration
    print("Creating inference configuration...")
    inference_config = InferenceConfig(
        num_warmup=100,  # Reduced for example
        num_samples=200,  # Reduced for example
        num_chains=1,
        method="svi",
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=500,
        guide_type="auto_normal",
    )

    # 5. Create models
    print("Creating models...")
    model1 = create_model(model_config1)
    model2 = create_model(model_config2)
    model3 = create_model(model_config3)

    # 6. Extract data
    u_obs = data_dict["X_unspliced"]
    s_obs = data_dict["X_spliced"]
    u_log_library = jnp.log(data_dict["u_lib_size"])
    s_log_library = jnp.log(data_dict["s_lib_size"])

    # Add batch dimension for the model
    u_obs = u_obs[jnp.newaxis, :, :]
    s_obs = s_obs[jnp.newaxis, :, :]
    u_log_library = u_log_library[jnp.newaxis, :]
    s_log_library = s_log_library[jnp.newaxis, :]

    # 7. Run inference for each model
    print("Running inference for model 1...")
    key, subkey1 = jax.random.split(key)
    _, inference_state1 = run_inference(
        model=model1,
        args=(),
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        config=inference_config,
        key=subkey1,
    )

    print("Running inference for model 2...")
    key, subkey2 = jax.random.split(key)
    _, inference_state2 = run_inference(
        model=model2,
        args=(),
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        config=inference_config,
        key=subkey2,
    )

    print("Running inference for model 3...")
    key, subkey3 = jax.random.split(key)
    _, inference_state3 = run_inference(
        model=model3,
        args=(),
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        config=inference_config,
        key=subkey3,
    )

    # 8. Compare models
    print("Comparing models...")
    key, subkey4 = jax.random.split(key)

    # Create a dictionary of models
    models = {
        "Model 1 (LogNormal + Poisson)": (model1, inference_state1),
        "Model 2 (Informative + Poisson)": (model2, inference_state2),
        "Model 3 (LogNormal + NegBin)": (model3, inference_state3),
    }

    # Compare models
    comparison_results = compare_models(
        models=models,
        args=(),
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        num_samples=100,  # Reduced for example
        key=subkey4,
    )

    # Print comparison results
    print("\nModel Comparison Results:")
    print("-" * 80)
    print(f"{'Model':<30} {'Log Likelihood':<15} {'WAIC':<10} {'LOO':<10} {'Weight':<10}")
    print("-" * 80)
    for model_name, metrics in comparison_results.items():
        log_likelihood = metrics.get('log_likelihood', float('nan'))
        waic = metrics.get('waic', float('nan'))
        loo = metrics.get('loo', float('nan'))
        weight = metrics.get('weight', float('nan'))
        print(f"{model_name:<30} {log_likelihood:<15.4f} {waic:<10.2f} {loo:<10.2f} {weight:<10.4f}")
    print("-" * 80)

    # 9. Select best model
    print("\nSelecting best model...")
    key, subkey5 = jax.random.split(key)

    # For demonstration purposes, if all metrics are infinite, just use the first model
    if all(jnp.isinf(metrics.get('waic', float('inf'))) for metrics in comparison_results.values()):
        print("All WAIC values are infinite. Using log likelihood for model selection.")
        # Use log likelihood instead
        best_model_name = max(
            comparison_results.keys(),
            key=lambda name: comparison_results[name].get('log_likelihood', float('-inf'))
        )
    else:
        # Use select_best_model function
        best_model_name, _ = select_best_model(
            models=models,
            args=(),
            kwargs={
                "u_obs": u_obs,
                "s_obs": s_obs,
                "u_log_library": u_log_library,
                "s_log_library": s_log_library
            },
            criterion="waic",  # Use WAIC for model selection
            num_samples=100,  # Reduced for example
            key=subkey5,
        )

    print(f"Selected model: {best_model_name}")

    # 10. Analyze posterior for the best model
    print("\nAnalyzing posterior for the best model...")
    best_model, best_inference_state = models[best_model_name]

    # Analyze posterior
    key, subkey6 = jax.random.split(key)
    # For demonstration purposes, we're not using the posterior_results
    # but in a real application, you would use them for further analysis
    _ = analyze_posterior(
        inference_state=best_inference_state,
        model=best_model,
        args=(),
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        num_samples=100,  # Reduced for example
        key=subkey6,
    )

    # 11. Store results in AnnData
    print("Storing results in AnnData...")
    posterior_samples = best_inference_state.posterior_samples
    velocity = compute_velocity(posterior_samples)

    # Flatten the velocity dictionary
    results = {}
    for key, value in velocity.items():
        results[key] = value

    # Add mean parameters
    results["alpha"] = jnp.mean(posterior_samples["alpha"], axis=0)
    results["beta"] = jnp.mean(posterior_samples["beta"], axis=0)
    results["gamma"] = jnp.mean(posterior_samples["gamma"], axis=0)

    # Add optional parameters if they exist in posterior_samples
    if "tau" in posterior_samples:
        results["latent_time"] = jnp.mean(posterior_samples["tau"], axis=0)

    adata_out = format_anndata_output(adata, results)

    # 12. Visualize results
    print("Visualizing results...")

    # Plot model comparison results
    plt.figure(figsize=(10, 6))
    model_names = list(comparison_results.keys())
    waics = [comparison_results[name]["waic"] for name in model_names]
    weights = [comparison_results[name]["weight"] for name in model_names]

    plt.subplot(1, 2, 1)
    plt.bar(model_names, waics)
    plt.title("WAIC Comparison")
    plt.ylabel("WAIC (lower is better)")
    plt.xticks(rotation=45, ha="right")

    plt.subplot(1, 2, 2)
    plt.bar(model_names, weights)
    plt.title("Model Weights")
    plt.ylabel("Weight")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("model_comparison_results.png")

    # Visualize velocity results
    if "clusters" in adata_out.obs.columns:
        sc.pl.umap(adata_out, color="clusters", title="Cell Clusters")
        import scvelo as scv
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")

    # 13. Save results
    from pathlib import Path
    output_path = Path("velocity_results_best_model.h5ad")
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
