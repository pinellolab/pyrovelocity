import torch
import pyro
import numpy as np
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata


RANDOM_SEED = 42
REPORTS_SAVE_PATH="reports/docs/posterior_predictive"

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


# Step 2: Generate synthetic data with known true parameters from prior
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

# Extract parameters for plotting functions
prior_parameter_samples = {}
if "true_parameters" in prior_predictive_adata.uns:
    for key, value in prior_predictive_adata.uns['true_parameters'].items():
        prior_parameter_samples[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

fig_prior = plot_prior_predictive_checks(
    model=model,
    prior_adata=prior_predictive_adata,
    prior_parameters=prior_parameter_samples,
    figsize=(7.5, 5.0),
    save_path=f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/sample_data",
    figure_name=f"posterior_predictive_check_sample_data_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=5,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,
)


# Step 3: Train model on prior predictive data
print(f"\n🎯 Training model on prior predictive data...")

trained_model = model.train(
    adata=prior_predictive_adata,
    max_epochs=1000,
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
posterior_parameter_samples = trained_model.generate_posterior_samples(
    adata=prior_predictive_adata,
    num_samples=30,
    seed=RANDOM_SEED,
    return_tensors=True
)

print(f"✅ Generated {len(posterior_parameter_samples)} types of posterior parameters")
for key in sorted(posterior_parameter_samples.keys()):
    param_shape = posterior_parameter_samples[key].shape if hasattr(posterior_parameter_samples[key], 'shape') else len(posterior_parameter_samples[key])
    print(f"    {key}: {param_shape}")


# Step 5: Generate posterior predictive data using posterior samples
print(f"\n📊 Generating posterior predictive data...")
print("  Using full posterior samples to generate synthetic data with uncertainty")

print(f"  📈 Posterior samples shape check:")
for key, value in posterior_parameter_samples.items():
    if hasattr(value, 'shape'):
        print(f"    {key}: {value.shape}")

# Generate posterior predictive data with uncertainty (no conversion needed!)
posterior_predictive_adata = trained_model.generate_predictive_samples(
    num_cells=prior_predictive_adata.n_obs,
    num_genes=prior_predictive_adata.n_vars,
    samples=posterior_parameter_samples,  # Use torch tensors directly
    return_format="anndata"
)

print("✅ Posterior predictive data generated:")
print_anndata(posterior_predictive_adata)

print("\n📊 Generate plots from posterior samples...")
print(f"✅ Using {len(posterior_parameter_samples)} types of posterior parameters")

# Copy UMAP coordinates for consistent visualization
print(f"\n🗺️ Copying UMAP coordinates from prior predictive data for consistent visualization...")
posterior_predictive_adata.obsm['X_umap'] = prior_predictive_adata.obsm['X_umap'].copy()
posterior_predictive_adata.uns['umap'] = prior_predictive_adata.uns['umap'].copy()

# Copy Leiden cluster labels from prior predictive data
print(f"📋 Copying Leiden cluster labels from prior predictive data...")
posterior_predictive_adata.obs['leiden'] = prior_predictive_adata.obs['leiden'].copy()
posterior_predictive_adata.uns['leiden'] = prior_predictive_adata.uns['leiden'].copy()

# Recompute PCA and neighbors for posterior predictive data
sc.pp.pca(posterior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(posterior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)

# Step 6: Generate posterior predictive check plots
print(f"\n🎨 Creating posterior predictive check plots...")

fig_posterior = plot_posterior_predictive_checks(
    model=trained_model,
    posterior_adata=posterior_predictive_adata,
    posterior_parameters=posterior_parameter_samples,
    figsize=(7.5, 5.0),
    save_path=f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}",
    figure_name=f"piecewise_activation_posterior_checks_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=5,
    observed_adata=prior_predictive_adata,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,  # Contains true parameters for validation
)

print(f"\n✅ Posterior predictive check workflow completed!")
print(f"📁 Plots saved to: {REPORTS_SAVE_PATH}")

print(f"\n🎯 Random seed used: {RANDOM_SEED}")
