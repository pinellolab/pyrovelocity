# AnnData Mutation Function Summary

This document summarizes the relationship between Python functions and their effects on AnnData fields. Functions are listed in the order they are executed within the preprocess, train, and postprocess tasks.

## Preprocessing Functions

### `pyrovelocity.tasks.preprocess.copy_raw_counts`

**Creates/Modifies:**

- `adata.layers["raw_unspliced"]` - Copies unspliced counts - `ndarray, n_obs x n_vars`
- `adata.layers["raw_spliced"]` - Copies spliced counts - `ndarray, n_obs x n_vars`
- `adata.obs["u_lib_size_raw"]` - Adds unspliced library size - `float64, n_obs`
- `adata.obs["s_lib_size_raw"]` - Adds spliced library size - `float64, n_obs`

### `scanpy.pp.filter_genes` (called in `preprocess_dataset`)

**Creates/Modifies:**

- Filters genes based on minimum counts - Reduces `n_vars`

### `scanpy.pp.normalize_total` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.X` - Normalizes count data - `ndarray, n_obs x n_vars`
- `adata.layers["spliced"]` - Normalizes spliced counts - `ndarray, n_obs x n_vars`
- `adata.layers["unspliced"]` - Normalizes unspliced counts - `ndarray, n_obs x n_vars`
- `adata.obs["initial_size_unspliced"]` - Stores original unspliced size - `float64, n_obs`
- `adata.obs["initial_size_spliced"]` - Stores original spliced size - `float64, n_obs`
- `adata.obs["initial_size"]` - Stores original total size - `float64, n_obs`
- `adata.obs["n_counts"]` - Stores normalized counts - `float64, n_obs`

### `scanpy.pp.log1p` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.X` - Log-transforms data - `ndarray, n_obs x n_vars`
- `adata.uns["log1p"]` - Adds log1p parameters - `dict`

### `scanpy.pp.pca` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.obsm["X_pca"]` - PCA coordinates - `ndarray, n_obs x n_pcs`
- `adata.varm["PCs"]` - PCA loadings - `ndarray, n_vars x n_pcs`
- `adata.uns["pca"]` - PCA parameters - `dict`

### `scanpy.pp.neighbors` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.uns["neighbors"]` - Neighbor parameters - `dict`
- `adata.obsp["distances"]` - Distance matrix - `csr_matrix, n_obs x n_obs`
- `adata.obsp["connectivities"]` - Connectivity matrix - `csr_matrix, n_obs x n_obs`

### `scvelo.pp.moments` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.layers["Ms"]` - Moments of spliced abundances - `ndarray, n_obs x n_vars`
- `adata.layers["Mu"]` - Moments of unspliced abundances - `ndarray, n_obs x n_vars`

### `scvelo.tl.recover_dynamics` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.var["fit_r2"]` - R² of fit - `float64, n_vars`
- `adata.var["fit_alpha"]` - Fitted alpha parameter - `float64, n_vars`
- `adata.var["fit_beta"]` - Fitted beta parameter - `float64, n_vars`
- `adata.var["fit_gamma"]` - Fitted gamma parameter - `float64, n_vars`
- `adata.var["fit_t_"]` - Fitted time parameter - `float64, n_vars`
- `adata.var["fit_scaling"]` - Fitted scaling parameter - `float64, n_vars`
- `adata.var["fit_std_u"]` - Fitted standard deviation for unspliced - `float64, n_vars`
- `adata.var["fit_std_s"]` - Fitted standard deviation for spliced - `float64, n_vars`
- `adata.var["fit_likelihood"]` - Fitted likelihood - `float64, n_vars`
- `adata.var["fit_u0"]` - Fitted initial unspliced - `float64, n_vars`
- `adata.var["fit_s0"]` - Fitted initial spliced - `float64, n_vars`
- `adata.var["fit_pval_steady"]` - Fitted p-value for steady state - `float64, n_vars`
- `adata.var["fit_steady_u"]` - Fitted steady state unspliced - `float64, n_vars`
- `adata.var["fit_steady_s"]` - Fitted steady state spliced - `float64, n_vars`
- `adata.var["fit_variance"]` - Fitted variance - `float64, n_vars`
- `adata.var["fit_alignment_scaling"]` - Fitted alignment scaling - `float64, n_vars`
- `adata.uns["recover_dynamics"]` - Recovery dynamics parameters - `dict`
- `adata.varm["loss"]` - Loss values - `ndarray, n_vars x n_loss_points`
- `adata.layers["fit_t"]` - Fitted time - `ndarray, n_obs x n_vars`
- `adata.layers["fit_tau"]` - Fitted tau - `ndarray, n_obs x n_vars`
- `adata.layers["fit_tau_"]` - Fitted tau_ - `ndarray, n_obs x n_vars`

### `scvelo.tl.velocity` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.var["velocity_genes"]` - Boolean indicating velocity genes - `bool, n_vars`
- `adata.uns["velocity_params"]` - Velocity parameters - `dict`
- `adata.layers["velocity"]` - Velocity vectors - `ndarray, n_obs x n_vars`
- `adata.layers["velocity_u"]` - Unspliced velocity - `ndarray, n_obs x n_vars`

### `scanpy.tl.leiden` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.obs["leiden"]` - Leiden clustering results - `category, n_obs`
- `adata.uns["leiden"]` - Leiden parameters - `dict`

### `scvelo.tl.velocity_graph` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.obs["velocity_self_transition"]` - Self-transition probabilities - `float32, n_obs`
- `adata.uns["velocity_graph"]` - Velocity graph - `csr_matrix, n_obs x n_obs`
- `adata.uns["velocity_graph_neg"]` - Negative velocity graph - `csr_matrix, n_obs x n_obs`

### `scvelo.tl.velocity_embedding` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.obsm["velocity_umap"]` - Embedded velocity vectors - `ndarray, n_obs x 2`

### `scvelo.tl.terminal_states` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.obs["root_cells"]` - Root cells - `float64, n_obs`
- `adata.obs["end_points"]` - End points - `float64, n_obs`

### `scvelo.tl.latent_time` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.obs["velocity_pseudotime"]` - Velocity pseudotime - `float64, n_obs`
- `adata.obs["latent_time"]` - Latent time - `float64, n_obs`

## Training Functions

### `pyrovelocity.models._velocity.PyroVelocity.setup_anndata`

**Creates/Modifies:**

- `adata.obs["u_lib_size"]` - Unspliced library size - `float64, n_obs`
- `adata.obs["s_lib_size"]` - Spliced library size - `float64, n_obs`
- `adata.obs["u_lib_size_mean"]` - Mean unspliced library size - `float64, 1`
- `adata.obs["s_lib_size_mean"]` - Mean spliced library size - `float64, 1`
- `adata.obs["u_lib_size_scale"]` - Scale of unspliced library size - `float64, 1`
- `adata.obs["s_lib_size_scale"]` - Scale of spliced library size - `float64, 1`
- `adata.obs["ind_x"]` - Cell indices - `int64, n_obs`
- `adata.uns["_scvi_uuid"]` - SCVI UUID - `str`
- `adata.uns["_scvi_manager_uuid"]` - SCVI manager UUID - `str`

## Postprocessing Functions

### `pyrovelocity.models._velocity.PyroVelocity.compute_statistics_from_posterior_samples`

This function calls `vector_field_uncertainty` and `compute_mean_vector_field` which modify the AnnData object.

### `pyrovelocity.analysis.analyze.vector_field_uncertainty`

**Creates/Modifies:**

- `adata.var["velocity_genes"]` - Boolean indicating velocity genes - `bool, n_vars`
- `adata.layers["spliced_pyro"]` - Posterior samples of spliced counts - `ndarray, n_obs x n_vars`
- `adata.layers["velocity_pyro"]` - Posterior samples of velocity - `ndarray, n_obs x n_vars`
- `adata.obsm["velocity_pyro_pca"]` - PCA-embedded velocity vectors (when basis="pca") - `ndarray, n_obs x n_pcs`
- `adata.obsm["velocity_pyro_umap"]` - UMAP-embedded velocity vectors (when basis="umap") - `ndarray, n_obs x 2`

### `pyrovelocity.analysis.analyze.compute_mean_vector_field`

**Creates/Modifies:**

- `adata.var["velocity_genes"]` - Boolean indicating velocity genes - `bool, n_vars`
- `adata.layers["spliced_pyro"]` - Mean of posterior samples of spliced counts - `ndarray, n_obs x n_vars`
- `adata.layers["velocity_pyro"]` - Mean of posterior samples of velocity - `ndarray, n_obs x n_vars`
- `adata.uns["velocity_pyro_graph"]` - Velocity graph - `csr_matrix, n_obs x n_obs`
- `adata.uns["velocity_pyro_graph_neg"]` - Negative velocity graph - `csr_matrix, n_obs x n_obs`
- `adata.uns["velocity_pyro_params"]` - Velocity parameters - `dict`
- `adata.obs["velocity_pyro_self_transition"]` - Self-transition probabilities - `float32, n_obs`
- `adata.obsm["velocity_pyro_umap"]` - UMAP-embedded velocity vectors - `ndarray, n_obs x 2`
