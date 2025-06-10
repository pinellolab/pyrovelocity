import torch
import pyro
import numpy as np
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata


RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 50)
print("🚀 Posterior Predictive Check Workflow")
print("=" * 50)

# Step 1: Create model
model = create_piecewise_activation_model()

print(f"✅ Created model: {model}")
print(f"Model components:")
print(f"  Dynamics: {model.dynamics_model}")
print(f"  Prior: {model.prior_model}")
print(f"  Likelihood: {model.likelihood_model}")
print(f"  Guide: {model.guide_model}")

# Step 2: Generate prior predictive data with known true parameters
print(f"\n📊 Generating prior predictive data (seed: {RANDOM_SEED})...")
prior_predictive_adata = model.generate_predictive_samples(
    num_cells=200,
    num_genes=100,
    num_samples=1,
    return_format="anndata"
)
print(f"\n🗺️ Computing UMAP and clustering for prior predictive data...")
sc.pp.pca(prior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
sc.tl.umap(prior_predictive_adata, random_state=RANDOM_SEED)
sc.tl.leiden(prior_predictive_adata, random_state=RANDOM_SEED)

print("✅ Prior predictive data generated:")
print_anndata(prior_predictive_adata)

# Step 3: Train model on prior predictive data
print(f"\n🎯 Training model on prior predictive data...")

trained_model = model.train(
    adata=prior_predictive_adata,
    max_epochs=2000,  # Reduced for faster testing
    learning_rate=0.01,
    early_stopping=True,
    early_stopping_patience=10,
    use_gpu="auto",
    seed=RANDOM_SEED
)
print("✅ Model training completed")

# Step 4: Generate posterior samples from trained model
print(f"\n🔬 Generating posterior samples from trained model...")

# Generate posterior samples using the trained model
posterior_samples = trained_model.generate_posterior_samples(
    adata=prior_predictive_adata,
    num_samples=100,  # Reduced for faster testing
    seed=RANDOM_SEED
)

print(f"✅ Generated {len(posterior_samples)} types of posterior parameters")
for key in sorted(posterior_samples.keys()):
    param_shape = posterior_samples[key].shape if hasattr(posterior_samples[key], 'shape') else len(posterior_samples[key])
    print(f"    {key}: {param_shape}")

# Step 5: Generate posterior predictive data using posterior samples
print(f"\n📊 Generating posterior predictive data...")
print("  Using posterior samples to generate synthetic data")

# Convert numpy arrays back to tensors for generate_predictive_samples
posterior_samples_tensors = {}
for key, value in posterior_samples.items():
    if isinstance(value, np.ndarray):
        # Take mean of samples for single parameter set
        if value.ndim > 1:
            posterior_samples_tensors[key] = torch.tensor(value.mean(axis=0))
        else:
            posterior_samples_tensors[key] = torch.tensor(value)
    else:
        posterior_samples_tensors[key] = value

# Generate posterior predictive data
posterior_predictive_adata = trained_model.generate_predictive_samples(
    num_cells=prior_predictive_adata.n_obs,
    num_genes=prior_predictive_adata.n_vars,
    samples=posterior_samples_tensors,
    return_format="anndata"
)

print("✅ Posterior predictive data generated:")
print_anndata(posterior_predictive_adata)

# Store fit parameters in the posterior predictive AnnData for plotting
print("\n📋 Storing fit parameters in AnnData...")
if "fit_parameters" not in posterior_predictive_adata.uns:
    fit_parameters = {}
    for key, value in posterior_samples.items():
        # Take mean of posterior samples
        if isinstance(value, np.ndarray) and value.ndim > 1:
            fit_parameters[key] = value.mean(axis=0)
        else:
            fit_parameters[key] = value

    # Store in AnnData
    posterior_predictive_adata.uns["fit_parameters"] = fit_parameters

print(f"✅ Stored {len(posterior_predictive_adata.uns['fit_parameters'])} fit parameters in AnnData")

# Extract parameters for plotting functions
posterior_parameter_samples = {}
for key, value in posterior_predictive_adata.uns["fit_parameters"].items():
    posterior_parameter_samples[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

# Add UMAP and clustering for visualization
print(f"\n🗺️ Computing UMAP and clustering for visualization...")
sc.pp.pca(adata=posterior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(adata=posterior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
# Use UMAP parameters associated with prior predictive data for consistent visualization
sc.tl.umap(adata=posterior_predictive_adata, **prior_predictive_adata.uns["umap"]["params"])
sc.tl.leiden(adata=posterior_predictive_adata, random_state=RANDOM_SEED)

# Step 6: Generate posterior predictive check plots
print(f"\n🎨 Creating posterior predictive check plots...")
REPORTS_SAVE_PATH="reports/docs/posterior_predictive"

fig_posterior = plot_posterior_predictive_checks(
    model=trained_model,
    posterior_adata=posterior_predictive_adata,
    posterior_parameters=posterior_parameter_samples,
    figsize=(7.5, 5.0),
    save_path=REPORTS_SAVE_PATH,
    figure_name=f"piecewise_activation_posterior_checks_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=5,
)

print(f"\n✅ Posterior predictive check workflow completed!")
print(f"📁 Plots saved to: {REPORTS_SAVE_PATH}")

print(f"\n🎯 Random seed used: {RANDOM_SEED}")
