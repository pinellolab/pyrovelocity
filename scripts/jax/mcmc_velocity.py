"""
End-to-end example of RNA velocity analysis using PyroVelocity JAX/NumPyro implementation with MCMC.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model
3. Performing MCMC inference
4. Analyzing and visualizing results

Note: MCMC initialization can be challenging for complex models. If you encounter
initialization errors, consider:
1. Using SVI first to find good initial parameters, then using MCMC
2. Simplifying the model (as done here by disabling latent time)
3. Using more informative priors
4. Increasing the number of initialization attempts
"""

import jax
import jax.numpy as jnp
import numpyro.infer as infer
import scanpy as sc
import scvelo as scv
import numpy as np
from importlib.resources import files

# Import core components
from pyrovelocity.models.jax.core.utils import create_key
from pyrovelocity.models.jax.core.state import InferenceConfig

# Import model creation
from pyrovelocity.models.jax.factory.factory import create_model

# Import data processing
from pyrovelocity.models.jax.data.anndata import prepare_anndata

# Import inference
from pyrovelocity.models.jax.inference.unified import run_inference

# Import analysis
from pyrovelocity.models.jax.inference.posterior import compute_velocity

# Import visualization
from pyrovelocity.models.jax.data.anndata import store_results as format_anndata_output

from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.models.jax.core.state import InferenceState

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
    # Import the necessary configuration classes
    from pyrovelocity.models.jax.factory.config import (
        ModelConfig,
        DynamicsFunctionConfig,
        PriorFunctionConfig,
        LikelihoodFunctionConfig,
        ObservationFunctionConfig,
        GuideFunctionConfig,
    )

    # Create the model configuration
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
        num_warmup=50,   # Further reduced for faster execution
        num_samples=100,  # Further reduced for faster execution
        num_chains=1,
        method="mcmc",  # Use MCMC instead of SVI
        # Use a lower learning rate and more epochs for initialization
        learning_rate=0.001,
        num_epochs=500,
        guide_type="auto_normal",  # Not used for MCMC directly, but may help with initialization
    )

    # 5. Create model
    print("Creating model...")
    model = create_model(model_config)

    # 6. First run SVI to get good initial parameters
    print("Running SVI to get initial parameters for MCMC...")
    key, subkey1 = jax.random.split(key)

    # Extract the data from data_dict and map to the expected parameter names
    u_obs = data_dict["X_unspliced"]
    s_obs = data_dict["X_spliced"]
    u_log_library = jnp.log(data_dict["u_lib_size"])
    s_log_library = jnp.log(data_dict["s_lib_size"])

    # Add batch dimension for the model
    u_obs = u_obs[jnp.newaxis, :, :]
    s_obs = s_obs[jnp.newaxis, :, :]
    u_log_library = u_log_library[jnp.newaxis, :]
    s_log_library = s_log_library[jnp.newaxis, :]

    # Create SVI config
    svi_config = InferenceConfig(
        method="svi",
        num_samples=200,
        num_warmup=100,
        num_chains=1,
        guide_type="auto_normal",
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=500,
    )

    # Run SVI inference to get initial parameters
    print("Running SVI for initialization...")
    _, svi_inference_state = run_inference(
        model=model,
        args=(),  # Empty tuple for positional args
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        config=svi_config,
        key=subkey1,
    )

    # Print SVI results
    svi_posterior_samples = svi_inference_state.posterior_samples
    print("SVI completed. Keys in posterior_samples:", list(svi_posterior_samples.keys()))

    # Now run MCMC with the SVI results as a starting point
    print("Running MCMC inference using SVI results as initialization...")
    key, subkey2 = jax.random.split(key)

    # Extract mean values from SVI posterior samples for initialization
    init_values = {}
    for param_name, param_samples in svi_posterior_samples.items():
        # Only use parameters that are part of the model (not computed values)
        if param_name in ["alpha", "beta", "gamma"]:
            # Use the mean of the SVI samples as the initial value
            init_values[param_name] = jnp.mean(param_samples, axis=0)

    print("Using SVI results for initialization with parameters:", list(init_values.keys()))

    # Create NUTS kernel with init_to_value initialization
    nuts_kernel = infer.NUTS(model)

    # Create MCMC object
    mcmc = infer.MCMC(
        nuts_kernel,
        num_warmup=inference_config.num_warmup,
        num_samples=inference_config.num_samples,
        num_chains=inference_config.num_chains,
        progress_bar=True,
    )

    # Run MCMC
    try:
        print("Running MCMC with SVI initialization...")
        # Pass init_values as init_params
        mcmc.run(
            subkey2,
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            init_params=init_values
        )

        # Extract posterior samples
        posterior_samples = mcmc.get_samples()

        # Create inference state (not used but kept for reference)
        _ = InferenceState(
            posterior_samples=posterior_samples,
            diagnostics={"accept_rate": mcmc.get_extra_fields().get("accept_prob", None)}
        )

        print("MCMC completed successfully!")
    except Exception as e:
        print(f"MCMC failed: {e}")
        print("Using SVI results instead...")

        # Use SVI results as fallback
        posterior_samples = svi_posterior_samples
        # We don't need to use the inference state directly

    # 7. Analyze results
    print("Analyzing results...")

    # Print the keys in posterior_samples to see what's available
    print("Keys in posterior_samples:", list(posterior_samples.keys()))

    # Use the default dynamics_fn (standard_dynamics_model) by not providing a second argument
    velocity = compute_velocity(posterior_samples)

    # 8. Store results in AnnData
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

    # Print the columns in the AnnData object to see what's available
    print("Columns in adata_out.obs:", list(adata_out.obs.columns))
    print("Keys in adata_out.uns:", list(adata_out.uns.keys()) if hasattr(adata_out, 'uns') and adata_out.uns is not None else "None")

    # Since we disabled latent time, we'll visualize other parameters
    print("Visualizing results...")

    # Add alpha, beta, gamma to var for visualization
    adata_out.var["alpha"] = np.array(jnp.mean(posterior_samples["alpha"], axis=0))
    adata_out.var["beta"] = np.array(jnp.mean(posterior_samples["beta"], axis=0))
    adata_out.var["gamma"] = np.array(jnp.mean(posterior_samples["gamma"], axis=0))

    # Visualize clusters
    if "clusters" in adata_out.obs.columns:
        print("Visualizing clusters...")
        sc.pl.umap(adata_out, color="clusters", title="Cell Clusters")

    # Check if 'clusters' exists in the AnnData object
    if 'clusters' in adata_out.obs.columns:
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(adata_out, basis="umap")

    # 10. Save results
    from pathlib import Path
    output_path = Path("velocity_results_mcmc.h5ad")
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()