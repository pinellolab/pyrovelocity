import torch
import pyro
import numpy as np
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_posterior_predictive_checks,
    plot_parameter_recovery_correlation,
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
    observed_adata=prior_predictive_adata,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,  # Contains true parameters for validation
)

# Step 7: Generate parameter recovery correlation analysis
print(f"\n📊 Creating parameter recovery correlation analysis...")

fig_recovery, recovery_metrics = plot_parameter_recovery_correlation(
    posterior_parameters=posterior_parameter_samples,
    true_parameters_adata=prior_predictive_adata,
    parameters_to_validate=["R_on", "gamma_star", "t_on_star", "delta_star"],
    figsize=(10, 8),
    save_path=REPORTS_SAVE_PATH,
    file_prefix="06",
    model=trained_model,
    default_fontsize=8
)

print(f"\n📈 Parameter Recovery Results:")
print(f"   Overall recovery quality: {recovery_metrics['summary']['overall_recovery_quality']}")
print(f"   Mean Pearson correlation: {recovery_metrics['summary']['mean_pearson_r']:.3f}")
print(f"   Mean R²: {recovery_metrics['summary']['mean_r_squared']:.3f}")

# Print individual parameter correlations
print(f"\n📋 Individual Parameter Correlations:")
for param_name in ["R_on", "gamma_star", "t_on_star", "delta_star"]:
    if param_name in recovery_metrics:
        r = recovery_metrics[param_name]['pearson_r']
        r2 = recovery_metrics[param_name]['r_squared']
        print(f"   {param_name}: r = {r:.3f}, R² = {r2:.3f}")

print(f"\n✅ Posterior predictive check workflow completed!")
print(f"📁 Plots saved to: {REPORTS_SAVE_PATH}")

print(f"\n🎯 Random seed used: {RANDOM_SEED}")
