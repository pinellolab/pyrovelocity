"""
Example of cross-validation for model selection using PyroVelocity JAX/NumPyro implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model
3. Performing cross-validation for model selection
4. Analyzing and visualizing results
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
from pyrovelocity.models.jax.comparison import cross_validate

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

    # 3. Create model configuration
    print("Creating model configuration...")
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )

    # 4. Create inference configuration
    print("Creating inference configuration...")
    inference_config = InferenceConfig(
        num_warmup=50,   # Further reduced for cross-validation
        num_samples=100,  # Further reduced for cross-validation
        num_chains=1,
        method="svi",
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=300,   # Reduced for cross-validation
        guide_type="auto_normal",
    )

    # 5. Create model
    print("Creating model...")
    model = create_model(model_config)

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

    # 7. Create data splits for cross-validation
    print("Creating data splits for cross-validation...")

    # For simplicity, we'll use a simple 2-fold cross-validation
    # In a real application, you would use more folds and more sophisticated splitting

    # Get the number of cells
    n_cells = u_obs.shape[1]

    # Create a random permutation of cell indices
    key, subkey = jax.random.split(key)
    cell_indices = jax.random.permutation(subkey, jnp.arange(n_cells))

    # Split into 2 folds
    fold_size = n_cells // 2
    fold1_indices = cell_indices[:fold_size]
    fold2_indices = cell_indices[fold_size:]

    # Create data splits
    data_splits = [
        # Fold 1: Train on fold 1, test on fold 2
        (
            (),  # Empty tuple for positional args
            {
                "u_obs": u_obs[:, fold1_indices, :],
                "s_obs": s_obs[:, fold1_indices, :],
                "u_log_library": u_log_library[:, fold1_indices],
                "s_log_library": s_log_library[:, fold1_indices]
            },
            (),  # Empty tuple for positional args
            {
                "u_obs": u_obs[:, fold2_indices, :],
                "s_obs": s_obs[:, fold2_indices, :],
                "u_log_library": u_log_library[:, fold2_indices],
                "s_log_library": s_log_library[:, fold2_indices]
            }
        ),
        # Fold 2: Train on fold 2, test on fold 1
        (
            (),  # Empty tuple for positional args
            {
                "u_obs": u_obs[:, fold2_indices, :],
                "s_obs": s_obs[:, fold2_indices, :],
                "u_log_library": u_log_library[:, fold2_indices],
                "s_log_library": s_log_library[:, fold2_indices]
            },
            (),  # Empty tuple for positional args
            {
                "u_obs": u_obs[:, fold1_indices, :],
                "s_obs": s_obs[:, fold1_indices, :],
                "u_log_library": u_log_library[:, fold1_indices],
                "s_log_library": s_log_library[:, fold1_indices]
            }
        )
    ]

    # 8. Define inference function for cross-validation
    def inference_fn(model, args, kwargs, key):
        """Run inference for cross-validation."""
        _, inference_state = run_inference(
            model=model,
            args=args,
            kwargs=kwargs,
            config=inference_config,
            key=key,
        )
        return inference_state

    # 9. Perform cross-validation
    print("Performing cross-validation...")
    key, subkey = jax.random.split(key)
    cv_results = cross_validate(
        model_fn=model,
        data_splits=data_splits,
        inference_fn=inference_fn,
        num_samples=50,  # Reduced for example
        key=subkey,
    )

    # 10. Print cross-validation results
    print("\nCross-Validation Results:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Values':<20}")
    print("-" * 80)
    for metric in ["log_likelihood", "waic", "loo"]:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in cv_results and std_key in cv_results:
            values_str = ', '.join([f"{val:.2f}" if not jnp.isinf(val) else "inf" for val in cv_results[metric]])
            if len(values_str) > 20:
                values_str = values_str[:17] + "..."
            print(f"{metric:<20} {cv_results[mean_key]:<10.4f} {cv_results[std_key]:<10.4f} {values_str:<20}")
    print("-" * 80)

    # 11. Visualize cross-validation results
    print("Visualizing cross-validation results...")

    plt.figure(figsize=(10, 6))

    # Plot log likelihood for each fold
    plt.subplot(1, 3, 1)
    plt.bar(range(len(cv_results["log_likelihood"])), cv_results["log_likelihood"])
    plt.title("Log Likelihood by Fold")
    plt.xlabel("Fold")
    plt.ylabel("Log Likelihood")

    # Plot WAIC for each fold
    plt.subplot(1, 3, 2)
    plt.bar(range(len(cv_results["waic"])), cv_results["waic"])
    plt.title("WAIC by Fold")
    plt.xlabel("Fold")
    plt.ylabel("WAIC (lower is better)")

    # Plot LOO for each fold
    plt.subplot(1, 3, 3)
    plt.bar(range(len(cv_results["loo"])), cv_results["loo"])
    plt.title("LOO by Fold")
    plt.xlabel("Fold")
    plt.ylabel("LOO (lower is better)")

    plt.tight_layout()
    plt.savefig("cross_validation_results.png")

    # 12. Run inference on the full dataset
    print("\nRunning inference on the full dataset...")
    key, subkey = jax.random.split(key)
    _, inference_state = run_inference(
        model=model,
        args=(),
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        config=inference_config,
        key=subkey,
    )

    # 13. Analyze results
    print("Analyzing results...")
    posterior_samples = inference_state.posterior_samples

    # Print the keys in posterior_samples to see what's available
    print("Keys in posterior_samples:", list(posterior_samples.keys()))

    # Compute velocity
    from pyrovelocity.models.jax.inference.posterior import compute_velocity
    velocity = compute_velocity(posterior_samples)

    # 14. Store results in AnnData
    print("Storing results in AnnData...")
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

    # 15. Visualize results
    print("Visualizing velocity results...")
    if "clusters" in adata_out.obs.columns:
        sc.pl.umap(adata_out, color="clusters", title="Cell Clusters")
        import scvelo as scv
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")

    # 16. Save results
    from pathlib import Path
    output_path = Path("velocity_results_cv.h5ad")
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
