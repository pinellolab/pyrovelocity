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
print("  Using full posterior samples to generate synthetic data with uncertainty")

# Convert numpy arrays back to tensors for generate_predictive_samples
# Keep all samples (don't average) to preserve posterior uncertainty
posterior_samples_tensors = {}
for key, value in posterior_samples.items():
    if isinstance(value, np.ndarray):
        posterior_samples_tensors[key] = torch.tensor(value)
    else:
        posterior_samples_tensors[key] = value

print(f"  📈 Posterior samples shape check:")
for key, value in posterior_samples_tensors.items():
    if hasattr(value, 'shape'):
        print(f"    {key}: {value.shape}")

# Generate posterior predictive data with uncertainty
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

# Check if we have uncertainty information
has_uncertainty = posterior_predictive_adata.uns.get("pyrovelocity", {}).get("has_posterior_uncertainty", False)
print(f"  📊 Posterior uncertainty available: {has_uncertainty}")

if "fit_parameters" not in posterior_predictive_adata.uns:
    fit_parameters = {}
    for key, value in posterior_samples.items():
        # Store mean for plotting compatibility, but keep full samples for uncertainty
        if isinstance(value, np.ndarray) and value.ndim > 1:
            fit_parameters[key] = value.mean(axis=0)
        else:
            fit_parameters[key] = value

    # Store in AnnData
    posterior_predictive_adata.uns["fit_parameters"] = fit_parameters

    # Also store full posterior samples for uncertainty analysis
    posterior_predictive_adata.uns["posterior_samples_full"] = posterior_samples

print(f"✅ Stored {len(posterior_predictive_adata.uns['fit_parameters'])} fit parameters in AnnData")
if has_uncertainty:
    print(f"✅ Stored full posterior samples for uncertainty analysis")

# Extract parameters for plotting functions (use full posterior samples)
if has_uncertainty and "posterior_samples_full" in posterior_predictive_adata.uns:
    print("  📊 Using full posterior samples for plotting")
    posterior_parameter_samples = {}
    for key, value in posterior_predictive_adata.uns["posterior_samples_full"].items():
        if isinstance(value, np.ndarray):
            posterior_parameter_samples[key] = torch.tensor(value)
        else:
            posterior_parameter_samples[key] = value
else:
    print("  📊 Using averaged parameters for plotting (fallback)")
    posterior_parameter_samples = {}
    for key, value in posterior_predictive_adata.uns["fit_parameters"].items():
        posterior_parameter_samples[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

# Copy UMAP coordinates and related data for consistent visualization
print(f"\n🗺️ Copying UMAP coordinates from prior predictive data for consistent visualization...")
# Copy UMAP coordinates directly to ensure identical coordinate system
posterior_predictive_adata.obsm['X_umap'] = prior_predictive_adata.obsm['X_umap'].copy()
# Copy PCA and neighbors data for completeness
posterior_predictive_adata.obsm['X_pca'] = prior_predictive_adata.obsm['X_pca'].copy()
posterior_predictive_adata.varm['PCs'] = prior_predictive_adata.varm['PCs'].copy()
posterior_predictive_adata.obsp['distances'] = prior_predictive_adata.obsp['distances'].copy()
posterior_predictive_adata.obsp['connectivities'] = prior_predictive_adata.obsp['connectivities'].copy()
# Copy metadata
posterior_predictive_adata.uns['pca'] = prior_predictive_adata.uns['pca'].copy()
posterior_predictive_adata.uns['neighbors'] = prior_predictive_adata.uns['neighbors'].copy()
posterior_predictive_adata.uns['umap'] = prior_predictive_adata.uns['umap'].copy()

# Copy Leiden cluster labels from prior predictive data to track cell movement
print(f"📋 Copying Leiden cluster labels from prior predictive data...")
posterior_predictive_adata.obs['leiden'] = prior_predictive_adata.obs['leiden'].copy()
posterior_predictive_adata.uns['leiden'] = prior_predictive_adata.uns['leiden'].copy()
print(f"✅ Preserved {len(posterior_predictive_adata.obs['leiden'].cat.categories)} Leiden clusters from prior predictive data")

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
    observed_adata=prior_predictive_adata,  # Use original prior predictive data as "observed"
)

print(f"\n✅ Posterior predictive check workflow completed!")
print(f"📁 Plots saved to: {REPORTS_SAVE_PATH}")

print(f"\n🎯 Random seed used: {RANDOM_SEED}")
