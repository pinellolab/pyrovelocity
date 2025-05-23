# AnnData Mutation Function Summary

This document summarizes the relationship between Python functions and their effects on AnnData fields. Functions are listed in the order they are executed within the preprocess, train, and postprocess tasks.

## Preprocessing Functions

### `pyrovelocity.tasks.preprocess.compute_and_plot_qc`

**Creates/Modifies:**

- `adata.var["mt"]` - Boolean flag for mitochondrial genes - `bool, n_vars`
- `adata.var["ribo"]` - Boolean flag for ribosomal genes - `bool, n_vars`

### `scanpy.pp.calculate_qc_metrics` (called in `compute_and_plot_qc`)

**Creates/Modifies:**

- `adata.obs["n_genes_by_counts"]` - Number of genes with counts for each cell - `int64/category, n_obs`
- `adata.obs["total_counts"]` - Total counts per cell - `float64, n_obs`
- `adata.obs["total_counts_mt"]` - Total counts in mitochondrial genes - `float64, n_obs`
- `adata.obs["pct_counts_mt"]` - Percentage of counts in mitochondrial genes - `float64, n_obs`
- `adata.obs["total_counts_ribo"]` - Total counts in ribosomal genes - `float64, n_obs`
- `adata.obs["pct_counts_ribo"]` - Percentage of counts in ribosomal genes - `float64, n_obs`
- `adata.var["n_cells_by_counts"]` - Number of cells with counts for each gene - `int64, n_vars`
- `adata.var["mean_counts"]` - Mean counts per gene - `float64, n_vars`
- `adata.var["pct_dropout_by_counts"]` - Percentage of cells with no counts for each gene - `float64, n_vars`
- `adata.var["total_counts"]` - Total counts per gene - `float64, n_vars`

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

### `scanpy.tl.umap` (called in `preprocess_dataset`)

**Creates/Modifies:**

- `adata.obsm["X_umap"]` - UMAP coordinates - `ndarray, n_obs x 2`
- `adata.uns["umap"]` - UMAP parameters - `dict`

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

## Example Initial and Final preprocessing states

### Initial State

```python
AnnData object with n_obs × n_vars = 100 × 300
    obs:
        true_t, float64, 94,
    var:
        true_t_, float64, 4,
        true_alpha, float64, 1,
        true_beta, float64, 1,
        true_gamma, float64, 1,
        true_scaling, float64, 1,
    layers:
        spliced, ndarray, 100 x 300,
        unspliced, ndarray, 100 x 300,
```

### Final expected state after preprocessing only

```python
AnnData object with n_obs × n_vars = 100 × 300
    obs:
        true_t, float64, 94,
        n_genes_by_counts, category, 37,
        total_counts, float64, 99,
        total_counts_mt, float64, 1,
        pct_counts_mt, float64, 1,
        total_counts_ribo, float64, 1,
        pct_counts_ribo, float64, 1,
        u_lib_size_raw, float64, 97,
        s_lib_size_raw, float64, 99,
        initial_size_unspliced, float64, 97,
        initial_size_spliced, float64, 99,
        initial_size, float64, 99,
        n_counts, float64, 5,
        leiden, category, 5,
        velocity_self_transition, float32, 78,
        root_cells, float64, 55,
        end_points, float64, 56,
        velocity_pseudotime, float64, 95,
        latent_time, float64, 78,
    var:
        true_t_, float64, 4,
        true_alpha, float64, 1,
        true_beta, float64, 1,
        true_gamma, float64, 1,
        true_scaling, float64, 1,
        mt, bool, 1,
        ribo, bool, 1,
        n_cells_by_counts, int64, 52,
        mean_counts, float64, 243,
        pct_dropout_by_counts, float64, 52,
        total_counts, float64, 243,
        fit_r2, float64, 300,
        fit_alpha, float64, 232,
        fit_beta, float64, 232,
        fit_gamma, float64, 232,
        fit_t_, float64, 231,
        fit_scaling, float64, 232,
        fit_std_u, float64, 232,
        fit_std_s, float64, 232,
        fit_likelihood, float64, 232,
        fit_u0, float64, 1,
        fit_s0, float64, 1,
        fit_pval_steady, float64, 232,
        fit_steady_u, float64, 232,
        fit_steady_s, float64, 232,
        fit_variance, float64, 232,
        fit_alignment_scaling, float64, 231,
        velocity_genes, bool, 2,
    uns:
        log1p, dict,
        pca, dict,
        neighbors, dict,
        recover_dynamics, dict,
        velocity_params, dict,
        umap, dict,
        leiden, dict,
        velocity_graph, csr_matrix, 100 x 100,
        velocity_graph_neg, csr_matrix, 100 x 100,
    obsm:
        X_pca, ndarray, 100 x 50,
        X_umap, ndarray, 100 x 2,
        velocity_umap, ndarray, 100 x 2,
    varm:
        PCs, ndarray, 300 x 50,
        loss, ndarray, 300 x 15,
    layers:
        spliced, ndarray, 100 x 300,
        unspliced, ndarray, 100 x 300,
        raw_unspliced, csr_matrix, 100 x 300,
        raw_spliced, csr_matrix, 100 x 300,
        Ms, ndarray, 100 x 300,
        Mu, ndarray, 100 x 300,
        fit_t, ndarray, 100 x 300,
        fit_tau, ndarray, 100 x 300,
        fit_tau_, ndarray, 100 x 300,
        velocity, ndarray, 100 x 300,
        velocity_u, ndarray, 100 x 300,
    obsp:
        distances, csr_matrix, 100 x 100,
        connectivities, csr_matrix, 100 x 100,
```

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

## Example Initial and Final training and postprocessing states

### Initial

```python
AnnData object with n_obs × n_vars = 100 × 300
    obs:
        true_t, float64, 94,
    var:
        true_t_, float64, 4,
        true_alpha, float64, 1,
        true_beta, float64, 1,
        true_gamma, float64, 1,
        true_scaling, float64, 1,
    layers:
        spliced, ndarray, 100 x 300,
        unspliced, ndarray, 100 x 300,
```

### Final state after training and postprocessing only

```python
AnnData object with n_obs × n_vars = 100 × 12
    obs:
        true_t, float64, 94,
        u_lib_size_raw, float64, 46,
        s_lib_size_raw, float64, 52,
        leiden, category, 8, [0, 1, 2, 3, 4, 5, 6, 7],
        u_lib_size, float64, 46,
        s_lib_size, float64, 52,
        u_lib_size_mean, float64, 1,
        s_lib_size_mean, float64, 1,
        u_lib_size_scale, float64, 1,
        s_lib_size_scale, float64, 1,
        ind_x, int64, 100,
        velocity_pyro_self_transition, float32, 99,
    var:
        true_t_, float64, 4,
        true_alpha, float64, 1,
        true_beta, float64, 1,
        true_gamma, float64, 1,
        true_scaling, float64, 1,
        velocity_genes, bool, 1,
    uns:
        _scvi_manager_uuid, str,
        _scvi_uuid, str,
        leiden, dict,
        neighbors, dict,
        pca, dict,
        umap, dict,
        velocity_pyro_graph, csr_matrix, 100 x 100,
        velocity_pyro_graph_neg, csr_matrix, 100 x 100,
        velocity_pyro_params, dict,
    obsm:
        X_pca, ndarray, 100 x 11,
        X_umap, ndarray, 100 x 2,
        velocity_pyro_pca, ndarray, 100 x 11,
        velocity_pyro_umap, ndarray, 100 x 2,
    varm:
        PCs, ndarray, 12 x 11,
    layers:
        raw_spliced, ndarray, 100 x 12,
        raw_unspliced, ndarray, 100 x 12,
        spliced, ndarray, 100 x 12,
        spliced_pyro, ndarray, 100 x 12,
        unspliced, ndarray, 100 x 12,
        velocity_pyro, ndarray, 100 x 12,
    obsp:
        connectivities, csr_matrix, 100 x 100,
        distances, csr_matrix, 100 x 100,
```

### Final state after complete pipeline (preprocessing + training + postprocessing)

```python
AnnData object with n_obs × n_vars = 100 × 300
    X: ndarray, 100 x 300,  # Log-transformed normalized data
    obs:
        true_t, float64, 94,
        n_genes_by_counts, category, 37,
        total_counts, float64, 99,
        total_counts_mt, float64, 1,
        pct_counts_mt, float64, 1,
        total_counts_ribo, float64, 1,
        pct_counts_ribo, float64, 1,
        u_lib_size_raw, float64, 97,
        s_lib_size_raw, float64, 99,
        initial_size_unspliced, float64, 97,
        initial_size_spliced, float64, 99,
        initial_size, float64, 99,
        n_counts, float64, 5,
        leiden, category, 5,
        velocity_self_transition, float32, 78,
        root_cells, float64, 55,
        end_points, float64, 56,
        velocity_pseudotime, float64, 95,
        latent_time, float64, 78,
        u_lib_size, float64, 97,
        s_lib_size, float64, 99,
        u_lib_size_mean, float64, 1,
        s_lib_size_mean, float64, 1,
        u_lib_size_scale, float64, 1,
        s_lib_size_scale, float64, 1,
        ind_x, int64, 100,
        velocity_pyro_self_transition, float32, 99,
    var:
        true_t_, float64, 4,
        true_alpha, float64, 1,
        true_beta, float64, 1,
        true_gamma, float64, 1,
        true_scaling, float64, 1,
        mt, bool, 1,
        ribo, bool, 1,
        n_cells_by_counts, int64, 52,
        mean_counts, float64, 243,
        pct_dropout_by_counts, float64, 52,
        total_counts, float64, 243,
        fit_r2, float64, 300,
        fit_alpha, float64, 232,
        fit_beta, float64, 232,
        fit_gamma, float64, 232,
        fit_t_, float64, 231,
        fit_scaling, float64, 232,
        fit_std_u, float64, 232,
        fit_std_s, float64, 232,
        fit_likelihood, float64, 232,
        fit_u0, float64, 1,
        fit_s0, float64, 1,
        fit_pval_steady, float64, 232,
        fit_steady_u, float64, 232,
        fit_steady_s, float64, 232,
        fit_variance, float64, 232,
        fit_alignment_scaling, float64, 231,
        velocity_genes, bool, 2,
    uns:
        log1p, dict,
        pca, dict,
        neighbors, dict,
        recover_dynamics, dict,
        velocity_params, dict,
        umap, dict,
        leiden, dict,
        velocity_graph, csr_matrix, 100 x 100,
        velocity_graph_neg, csr_matrix, 100 x 100,
        _scvi_uuid, str,
        _scvi_manager_uuid, str,
        velocity_pyro_graph, csr_matrix, 100 x 100,
        velocity_pyro_graph_neg, csr_matrix, 100 x 100,
        velocity_pyro_params, dict,
    obsm:
        X_pca, ndarray, 100 x 50,
        X_umap, ndarray, 100 x 2,
        velocity_umap, ndarray, 100 x 2,
        velocity_pyro_pca, ndarray, 100 x 50,
        velocity_pyro_umap, ndarray, 100 x 2,
    varm:
        PCs, ndarray, 300 x 50,
        loss, ndarray, 300 x 15,
    layers:
        spliced, ndarray, 100 x 300,
        unspliced, ndarray, 100 x 300,
        raw_unspliced, csr_matrix, 100 x 300,
        raw_spliced, csr_matrix, 100 x 300,
        Ms, ndarray, 100 x 300,
        Mu, ndarray, 100 x 300,
        fit_t, ndarray, 100 x 300,
        fit_tau, ndarray, 100 x 300,
        fit_tau_, ndarray, 100 x 300,
        velocity, ndarray, 100 x 300,
        velocity_u, ndarray, 100 x 300,
        spliced_pyro, ndarray, 100 x 300,
        velocity_pyro, ndarray, 100 x 300,
    obsp:
        distances, csr_matrix, 100 x 100,
        connectivities, csr_matrix, 100 x 100,
```

### Final state for our pancreas_50_7 postprocessed data fixture

```python
AnnData object with n_obs × n_vars = 50 × 7
    obs:
        clusters_coarse, category, 5, [Ductal, Endocrine, Ngn3 high EP, Ngn3 low EP Pre-endocrine],
        clusters, category, 7, [Alpha, Beta, Ductal, Epsilon, Ngn3 high EP, Ngn3 low EP, Pre-endocrine],
        S_score, float64, 50,
        G2M_score, float64, 50,
        n_genes_by_counts, int64, 49,
        total_counts, float64, 48,
        total_counts_mt, float64, 30,
        pct_counts_mt, float64, 50,
        total_counts_ribo, float64, 50,
        pct_counts_ribo, float64, 50,
        u_lib_size_raw, float64, 50,
        s_lib_size_raw, float64, 48,
        gcs, float64, 50,
        cytotrace, float64, 50,
        counts, float64, 50,
        initial_size_unspliced, float64, 50,
        initial_size_spliced, float64, 48,
        initial_size, float64, 48,
        n_counts, float64, 50,
        leiden, category, 4, [0, 1, 2, 3],
        velocity_self_transition, float64, 43,
        u_lib_size, float64, 50,
        s_lib_size, float64, 48,
        u_lib_size_mean, float64, 1,
        s_lib_size_mean, float64, 1,
        u_lib_size_scale, float64, 1,
        s_lib_size_scale, float64, 1,
        ind_x, int64, 50,
        cell_time, float64, 50,
        1-Cytotrace, float64, 50,
        velocity_pyro_self_transition, float64, 41,
    var:
        highly_variable_genes, category, 1, [True],
        mt, bool, 1,
        ribo, bool, 1,
        n_cells_by_counts, int64, 7,
        mean_counts, float64, 7,
        pct_dropout_by_counts, float64, 7,
        total_counts, float64, 7,
        cytotrace, bool, 1,
        cytotrace_corrs, float64, 7,
        fit_r2, float64, 7,
        fit_alpha, float64, 7,
        fit_beta, float64, 7,
        fit_gamma, float64, 7,
        fit_t_, float64, 7,
        fit_scaling, float64, 7,
        fit_std_u, float64, 7,
        fit_std_s, float64, 7,
        fit_likelihood, float64, 7,
        fit_u0, float64, 1,
        fit_s0, float64, 1,
        fit_pval_steady, float64, 7,
        fit_steady_u, float64, 7,
        fit_steady_s, float64, 7,
        fit_variance, float64, 7,
        fit_alignment_scaling, float64, 7,
        velocity_genes, bool, 1,
    uns:
        _scvi_manager_uuid, str,
        _scvi_uuid, str,
        clusters_coarse_colors, ndarray, 5,
        clusters_colors, ndarray, 7,
        day_colors, ndarray, 1,
        leiden, dict,
        log1p, dict,
        neighbors, dict,
        pca, dict,
        recover_dynamics, dict,
        velocity_graph, ndarray, 50 x 50,
        velocity_graph_neg, ndarray, 50 x 50,
        velocity_params, dict,
        velocity_pyro_graph, list,
        velocity_pyro_graph_neg, list,
        velocity_pyro_params, dict,
    obsm:
        X_pca, ndarray, 50 x 6,
        X_umap, ndarray, 50 x 2,
        velocity_pyro_pca, ndarray, 50 x 6,
        velocity_pyro_umap, ndarray, 50 x 2,
        velocity_umap, ndarray, 50 x 2,
    varm:
        PCs, ndarray, 7 x 6,
        loss, ndarray, 7 x 15,
    layers:
        Ms, ndarray, 50 x 7,
        Mu, ndarray, 50 x 7,
        fit_t, ndarray, 50 x 7,
        fit_tau, ndarray, 50 x 7,
        fit_tau_, ndarray, 50 x 7,
        raw_spliced, ndarray, 50 x 7,
        raw_unspliced, ndarray, 50 x 7,
        spliced, ndarray, 50 x 7,
        spliced_pyro, ndarray, 50 x 7,
        unspliced, ndarray, 50 x 7,
        velocity, ndarray, 50 x 7,
        velocity_pyro, ndarray, 50 x 7,
        velocity_u, ndarray, 50 x 7,
    obsp:
        connectivities, ndarray, 50 x 50,
        distances, ndarray, 50 x 50,
```

which can be loaded using

```python
from importlib.resources import files

from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.utils import print_anndata, anndata_string, print_string_diff

# see `src/pyrovelocity/tests/fixtures/get_fixture_hashes.py` to update fixture hashes
FIXTURE_HASHES = {
    "preprocessed_pancreas_50_7.json": "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae",
    "trained_pancreas_50_7.json": "8c575d9de0430003b469b9cc9850171914a4fe1f0ae655fe0146f81af34abd04",
    "postprocessed_pancreas_50_7.json": "d50813ad23e4ae1c34f483547a7d8351fdfa94c805098caf99b8864eba8892ef",
    "larry_multilineage_50_6.json": "227a025f340e9ead0779abf8349e6c2a9774301b50e13f1c6d9f3f96001dfe73",
    "preprocessed_larry_multilineage_50_6.json": "61d3da04b5de323d3e0fd0bfd6218281c76e58f2d5271d52247f2f3218f1b1a2",
    "trained_larry_multilineage_50_6.json": "c6338e64b437e8b7a82f245729e585dc4fa7f11cd428777716e84fc6c46603f8",
    "postprocessed_larry_multilineage_50_6.json": "a8aeec31939a8d1b93577e5bf3d4f747d69cd4564bd87af54ec800903aaa25a6",
}


def adata_preprocessed_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["preprocessed_pancreas_50_7.json"],
    )


def adata_trained_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "trained_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["trained_pancreas_50_7.json"],
    )


def adata_postprocessed_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "postprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["postprocessed_pancreas_50_7.json"],
    )

adata_pp = adata_preprocessed_pancreas_50_7()
adata_tr = adata_trained_pancreas_50_7()
adata_pt = adata_postprocessed_pancreas_50_7()

print_string_diff(
    text1=anndata_string(adata_pp),
    text2=anndata_string(adata_tr),
    diff_title="trained vs preprocessed",
)
print_string_diff(
    text1=anndata_string(adata_tr),
    text2=anndata_string(adata_pt),
    diff_title="postprocessed vs trained",
)
print_string_diff(
    text1=anndata_string(adata_pp),
    text2=anndata_string(adata_pt),
    diff_title="preprocessed vs postprocessed",
)
print_anndata(adata_pt)
```

demonstrating that training and postprocessing adds or updates

```python
╭─────────── preprocessed vs postprocessed ────────────╮
│                                                      │
│ ---                                                  │
│ +++                                                  │
│ @@ -22,6 +22,16 @@                                   │
│          n_counts, float64, 50,                      │
│          leiden, category, 4, [0, 1, 2, 3],          │
│          velocity_self_transition, float64, 43,      │
│ +        u_lib_size, float64, 50,                    │
│ +        s_lib_size, float64, 48,                    │
│ +        u_lib_size_mean, float64, 1,                │
│ +        s_lib_size_mean, float64, 1,                │
│ +        u_lib_size_scale, float64, 1,               │
│ +        s_lib_size_scale, float64, 1,               │
│ +        ind_x, int64, 50,                           │
│ +        cell_time, float64, 50,                     │
│ +        1-Cytotrace, float64, 50,                   │
│ +        velocity_pyro_self_transition, float64, 41, │
│      var:                                            │
│          highly_variable_genes, category, 1, [True], │
│          mt, bool, 1,                                │
│ @@ -48,8 +58,10 @@                                   │
│          fit_steady_s, float64, 7,                   │
│          fit_variance, float64, 7,                   │
│          fit_alignment_scaling, float64, 7,          │
│ -        velocity_genes, bool, 2,                    │
│ +        velocity_genes, bool, 1,                    │
│      uns:                                            │
│ +        _scvi_manager_uuid, str,                    │
│ +        _scvi_uuid, str,                            │
│          clusters_coarse_colors, ndarray, 5,         │
│          clusters_colors, ndarray, 7,                │
│          day_colors, ndarray, 1,                     │
│ @@ -61,11 +73,17 @@                                  │
│          velocity_graph, ndarray, 50 x 50,           │
│          velocity_graph_neg, ndarray, 50 x 50,       │
│          velocity_params, dict,                      │
│ +        velocity_pyro_graph, list,                  │
│ +        velocity_pyro_graph_neg, list,              │
│ +        velocity_pyro_params, dict,                 │
│      obsm:                                           │
│ -        X_pca, ndarray, 50 x 50,                    │
│ +        X_pca, ndarray, 50 x 6,                     │
│          X_umap, ndarray, 50 x 2,                    │
│ +        velocity_pyro_pca, ndarray, 50 x 6,         │
│ +        velocity_pyro_umap, ndarray, 50 x 2,        │
│          velocity_umap, ndarray, 50 x 2,             │
│      varm:                                           │
│ +        PCs, ndarray, 7 x 6,                        │
│          loss, ndarray, 7 x 15,                      │
│      layers:                                         │
│          Ms, ndarray, 50 x 7,                        │
│ @@ -76,8 +94,10 @@                                   │
│          raw_spliced, ndarray, 50 x 7,               │
│          raw_unspliced, ndarray, 50 x 7,             │
│          spliced, ndarray, 50 x 7,                   │
│ +        spliced_pyro, ndarray, 50 x 7,              │
│          unspliced, ndarray, 50 x 7,                 │
│          velocity, ndarray, 50 x 7,                  │
│ +        velocity_pyro, ndarray, 50 x 7,             │
│          velocity_u, ndarray, 50 x 7,                │
│      obsp:                                           │
│          connectivities, ndarray, 50 x 50,           │
│                                                      │
╰──────────────────────────────────────────────────────╯
```
