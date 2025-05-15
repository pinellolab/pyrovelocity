# Preprocessing AnnData state mutation summary

```python
[ins] In [1]:         >>> from pathlib import Path
         ...:         >>> from pyrovelocity.tasks.data import download_dataset
         ...:         >>> from pyrovelocity.tasks.preprocess import preprocess_dataset
         ...:         >>> tmpdir = None
         ...:         >>> try:
         ...:         >>>     tmp = getfixture("tmp_path")
         ...:         >>> except NameError:
         ...:         >>>     import tempfile
         ...:         >>>     tmpdir = tempfile.TemporaryDirectory()
         ...:         >>>     tmp = tmpdir.name
         ...:         >>> simulated_dataset_path = download_dataset(
         ...:         ...   'simulated',
         ...:         ...   str(tmp) + '/data/external',
         ...:         ...   'simulate',
         ...:         ...   n_obs=100,
         ...:         ...   n_vars=300,
         ...:         ... )
         ...:         >>> preprocess_dataset(
         ...:         ...     data_set_name="simulated",
         ...:         ...     adata=simulated_dataset_path,
         ...:         ... )

[11:15:52] INFO     pyrovelocity.tasks.data

                    Verifying existence of path for downloaded data: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpv0rqbpfb/data/external

           INFO     pyrovelocity.tasks.data

                    Verifying simulated data:
                      data will be stored in /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpv0rqbpfb/data/external/simulated.h5ad

           INFO     pyrovelocity.tasks.data Attempting to download simulated data...
           INFO     pyrovelocity.tasks.data Generating simulated data from simulation...
           INFO     pyrovelocity.utils
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
                            unspliced, ndarray, 100 x 300,
                            spliced, ndarray, 100 x 300,
           INFO     pyrovelocity.tasks.data Successfully downloaded /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpv0rqbpfb/data/external/simulated.h5ad
           INFO     pyrovelocity.tasks.preprocess Reset random state from seed: 99
           INFO     pyrovelocity.utils Reading input file: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpv0rqbpfb/data/external/simulated.h5ad
           INFO     pyrovelocity.utils
                    Successfully read input file: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpv0rqbpfb/data/external/simulated.h5ad
                    SHA-256 hash: 68ae73fee164d39b30d91cfe9de48518de6c1dc0a78d0ba2104d48d2d646bb1b

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
           INFO     pyrovelocity.tasks.preprocess

                    Verifying existence of path for:

                      preprocessing reports: reports/processed/simulated

           INFO     pyrovelocity.tasks.preprocess

                    Verifying existence of path for:

                      processed data: data/processed

           INFO     pyrovelocity.tasks.preprocess

                    Preprocessing simulated data :

                      from: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpv0rqbpfb/data/external/simulated.h5ad
                      to processed: data/processed/simulated_processed.h5ad

╭────────── MT/Ribo flags diff ───────────╮
│                                         │
│ ---                                     │
│ +++                                     │
│ @@ -8,6 +8,8 @@                         │
│          true_beta, float64, 1,         │
│          true_gamma, float64, 1,        │
│          true_scaling, float64, 1,      │
│ +        mt, bool, 1,                   │
│ +        ribo, bool, 1,                 │
│      layers:                            │
│          spliced, ndarray, 100 x 300,   │
│          unspliced, ndarray, 100 x 300, │
│                                         │
╰─────────────────────────────────────────╯
╭────────── QC metrics calculation diff ──────────╮
│                                                 │
│ ---                                             │
│ +++                                             │
│ @@ -2,6 +2,12 @@                                │
│  AnnData object with n_obs × n_vars = 100 × 300 │
│      obs:                                       │
│          true_t, float64, 94,                   │
│ +        n_genes_by_counts, int64, 37,          │
│ +        total_counts, float64, 99,             │
│ +        total_counts_mt, float64, 1,           │
│ +        pct_counts_mt, float64, 1,             │
│ +        total_counts_ribo, float64, 1,         │
│ +        pct_counts_ribo, float64, 1,           │
│      var:                                       │
│          true_t_, float64, 4,                   │
│          true_alpha, float64, 1,                │
│ @@ -10,6 +16,10 @@                              │
│          true_scaling, float64, 1,              │
│          mt, bool, 1,                           │
│          ribo, bool, 1,                         │
│ +        n_cells_by_counts, int64, 52,          │
│ +        mean_counts, float64, 243,             │
│ +        pct_dropout_by_counts, float64, 52,    │
│ +        total_counts, float64, 243,            │
│      layers:                                    │
│          spliced, ndarray, 100 x 300,           │
│          unspliced, ndarray, 100 x 300,         │
│                                                 │
╰─────────────────────────────────────────────────╯
╭───────── Quality control metrics diff ──────────╮
│                                                 │
│ ---                                             │
│ +++                                             │
│ @@ -2,12 +2,24 @@                               │
│  AnnData object with n_obs × n_vars = 100 × 300 │
│      obs:                                       │
│          true_t, float64, 94,                   │
│ +        n_genes_by_counts, int64, 37,          │
│ +        total_counts, float64, 99,             │
│ +        total_counts_mt, float64, 1,           │
│ +        pct_counts_mt, float64, 1,             │
│ +        total_counts_ribo, float64, 1,         │
│ +        pct_counts_ribo, float64, 1,           │
│      var:                                       │
│          true_t_, float64, 4,                   │
│          true_alpha, float64, 1,                │
│          true_beta, float64, 1,                 │
│          true_gamma, float64, 1,                │
│          true_scaling, float64, 1,              │
│ +        mt, bool, 1,                           │
│ +        ribo, bool, 1,                         │
│ +        n_cells_by_counts, int64, 52,          │
│ +        mean_counts, float64, 243,             │
│ +        pct_dropout_by_counts, float64, 52,    │
│ +        total_counts, float64, 243,            │
│      layers:                                    │
│          spliced, ndarray, 100 x 300,           │
│          unspliced, ndarray, 100 x 300,         │
│                                                 │
╰─────────────────────────────────────────────────╯
[11:15:53] INFO     pyrovelocity.tasks.preprocess generating data/processed/simulated_processed.h5ad ...
           INFO     pyrovelocity.tasks.preprocess 'raw_unspliced' key added raw unspliced counts to adata.layers
           INFO     pyrovelocity.tasks.preprocess 'raw_spliced' key added raw spliced counts added to adata.layers
           INFO     pyrovelocity.tasks.preprocess 'u_lib_size_raw' key added unspliced library size to adata.obs, total: 151561.0
           INFO     pyrovelocity.tasks.preprocess 's_lib_size_raw' key added spliced library size to adata.obs, total: 162041.0
╭─────────── Copy raw counts diff ────────────╮
│                                             │
│ ---                                         │
│ +++                                         │
│ @@ -8,6 +8,8 @@                             │
│          pct_counts_mt, float64, 1,         │
│          total_counts_ribo, float64, 1,     │
│          pct_counts_ribo, float64, 1,       │
│ +        u_lib_size_raw, float64, 97,       │
│ +        s_lib_size_raw, float64, 99,       │
│      var:                                   │
│          true_t_, float64, 4,               │
│          true_alpha, float64, 1,            │
│ @@ -23,3 +25,5 @@                           │
│      layers:                                │
│          spliced, ndarray, 100 x 300,       │
│          unspliced, ndarray, 100 x 300,     │
│ +        raw_unspliced, ndarray, 100 x 300, │
│ +        raw_spliced, ndarray, 100 x 300,   │
│                                             │
╰─────────────────────────────────────────────╯
Normalized count data: X, spliced, unspliced.
Skip filtering by dispersion since number of variables are less than `n_top_genes`.
╭────────── Filter and normalize diff ──────────╮
│                                               │
│ ---                                           │
│ +++                                           │
│ @@ -10,6 +10,10 @@                            │
│          pct_counts_ribo, float64, 1,         │
│          u_lib_size_raw, float64, 97,         │
│          s_lib_size_raw, float64, 99,         │
│ +        initial_size_unspliced, float64, 97, │
│ +        initial_size_spliced, float64, 99,   │
│ +        initial_size, float64, 99,           │
│ +        n_counts, float64, 5,                │
│      var:                                     │
│          true_t_, float64, 4,                 │
│          true_alpha, float64, 1,              │
│ @@ -22,6 +26,8 @@                             │
│          mean_counts, float64, 243,           │
│          pct_dropout_by_counts, float64, 52,  │
│          total_counts, float64, 243,          │
│ +    uns:                                     │
│ +        log1p, dict,                         │
│      layers:                                  │
│          spliced, ndarray, 100 x 300,         │
│          unspliced, ndarray, 100 x 300,       │
│                                               │
╰───────────────────────────────────────────────╯
[11:15:54] WARNING  pyrovelocity.tasks.preprocess adata.n_vars: 300 < n_top_genes: 2000
                    for data_set_name: simulated and min_shared_counts: 30

           INFO     pyrovelocity.tasks.preprocess adata.shape before filtering: (100, 300)
           INFO     pyrovelocity.tasks.preprocess adata.shape after filtering: (100, 300)
╭───────── High US genes filtering diff ─────────╮
│                                                │
│ ---                                            │
│ +++                                            │
│ @@ -31,5 +31,5 @@                              │
│      layers:                                   │
│          spliced, ndarray, 100 x 300,          │
│          unspliced, ndarray, 100 x 300,        │
│ -        raw_unspliced, ndarray, 100 x 300,    │
│ -        raw_spliced, ndarray, 100 x 300,      │
│ +        raw_unspliced, csr_matrix, 100 x 300, │
│ +        raw_spliced, csr_matrix, 100 x 300,   │
│                                                │
╰────────────────────────────────────────────────╯
╭─────────────── PCA diff ────────────────╮
│                                         │
│ ---                                     │
│ +++                                     │
│ @@ -28,6 +28,11 @@                      │
│          total_counts, float64, 243,    │
│      uns:                               │
│          log1p, dict,                   │
│ +        pca, dict,                     │
│ +    obsm:                              │
│ +        X_pca, ndarray, 100 x 50,      │
│ +    varm:                              │
│ +        PCs, ndarray, 300 x 50,        │
│      layers:                            │
│          spliced, ndarray, 100 x 300,   │
│          unspliced, ndarray, 100 x 300, │
│                                         │
╰─────────────────────────────────────────╯
╭──────────────── Neighbors diff ─────────────────╮
│                                                 │
│ ---                                             │
│ +++                                             │
│ @@ -29,6 +29,7 @@                               │
│      uns:                                       │
│          log1p, dict,                           │
│          pca, dict,                             │
│ +        neighbors, dict,                       │
│      obsm:                                      │
│          X_pca, ndarray, 100 x 50,              │
│      varm:                                      │
│ @@ -38,3 +39,6 @@                               │
│          unspliced, ndarray, 100 x 300,         │
│          raw_unspliced, csr_matrix, 100 x 300,  │
│          raw_spliced, csr_matrix, 100 x 300,    │
│ +    obsp:                                      │
│ +        distances, csr_matrix, 100 x 100,      │
│ +        connectivities, csr_matrix, 100 x 100, │
│                                                 │
╰─────────────────────────────────────────────────╯
computing moments based on connectivities
    finished (0:00:00) --> added
    'Ms' and 'Mu', moments of un/spliced abundances (adata.layers)
╭───────────────── Moments diff ──────────────────╮
│                                                 │
│ ---                                             │
│ +++                                             │
│ @@ -39,6 +39,8 @@                               │
│          unspliced, ndarray, 100 x 300,         │
│          raw_unspliced, csr_matrix, 100 x 300,  │
│          raw_spliced, csr_matrix, 100 x 300,    │
│ +        Ms, ndarray, 100 x 300,                │
│ +        Mu, ndarray, 100 x 300,                │
│      obsp:                                      │
│          distances, csr_matrix, 100 x 100,      │
│          connectivities, csr_matrix, 100 x 100, │
│                                                 │
╰─────────────────────────────────────────────────╯
recovering dynamics (using 16/16 cores)
  0%|          | 0/232 [00:00<?, ?gene/s]
    finished (0:00:04) --> added
    'fit_pars', fitted parameters for splicing dynamics (adata.var)
╭───────────── Recover dynamics diff ─────────────╮
│                                                 │
│ ---                                             │
│ +++                                             │
│ @@ -26,14 +26,32 @@                             │
│          mean_counts, float64, 243,             │
│          pct_dropout_by_counts, float64, 52,    │
│          total_counts, float64, 243,            │
│ +        fit_r2, float64, 300,                  │
│ +        fit_alpha, float64, 232,               │
│ +        fit_beta, float64, 232,                │
│ +        fit_gamma, float64, 232,               │
│ +        fit_t_, float64, 231,                  │
│ +        fit_scaling, float64, 232,             │
│ +        fit_std_u, float64, 232,               │
│ +        fit_std_s, float64, 232,               │
│ +        fit_likelihood, float64, 232,          │
│ +        fit_u0, float64, 1,                    │
│ +        fit_s0, float64, 1,                    │
│ +        fit_pval_steady, float64, 232,         │
│ +        fit_steady_u, float64, 232,            │
│ +        fit_steady_s, float64, 232,            │
│ +        fit_variance, float64, 232,            │
│ +        fit_alignment_scaling, float64, 231,   │
│      uns:                                       │
│          log1p, dict,                           │
│          pca, dict,                             │
│          neighbors, dict,                       │
│ +        recover_dynamics, dict,                │
│      obsm:                                      │
│          X_pca, ndarray, 100 x 50,              │
│      varm:                                      │
│          PCs, ndarray, 300 x 50,                │
│ +        loss, ndarray, 300 x 15,               │
│      layers:                                    │
│          spliced, ndarray, 100 x 300,           │
│          unspliced, ndarray, 100 x 300,         │
│ @@ -41,6 +59,9 @@                               │
│          raw_spliced, csr_matrix, 100 x 300,    │
│          Ms, ndarray, 100 x 300,                │
│          Mu, ndarray, 100 x 300,                │
│ +        fit_t, ndarray, 100 x 300,             │
│ +        fit_tau, ndarray, 100 x 300,           │
│ +        fit_tau_, ndarray, 100 x 300,          │
│      obsp:                                      │
│          distances, csr_matrix, 100 x 100,      │
│          connectivities, csr_matrix, 100 x 100, │
│                                                 │
╰─────────────────────────────────────────────────╯
computing velocities
    finished (0:00:00) --> added
    'velocity', velocity vectors for each individual cell (adata.layers)
╭───────────────── Velocity diff ─────────────────╮
│                                                 │
│ ---                                             │
│ +++                                             │
│ @@ -2,7 +2,7 @@                                 │
│  AnnData object with n_obs × n_vars = 100 × 300 │
│      obs:                                       │
│          true_t, float64, 94,                   │
│ -        n_genes_by_counts, int64, 37,          │
│ +        n_genes_by_counts, category, 37,       │
│          total_counts, float64, 99,             │
│          total_counts_mt, float64, 1,           │
│          pct_counts_mt, float64, 1,             │
│ @@ -42,11 +42,13 @@                             │
│          fit_steady_s, float64, 232,            │
│          fit_variance, float64, 232,            │
│          fit_alignment_scaling, float64, 231,   │
│ +        velocity_genes, bool, 2,               │
│      uns:                                       │
│          log1p, dict,                           │
│          pca, dict,                             │
│          neighbors, dict,                       │
│          recover_dynamics, dict,                │
│ +        velocity_params, dict,                 │
│      obsm:                                      │
│          X_pca, ndarray, 100 x 50,              │
│      varm:                                      │
│ @@ -62,6 +64,8 @@                               │
│          fit_t, ndarray, 100 x 300,             │
│          fit_tau, ndarray, 100 x 300,           │
│          fit_tau_, ndarray, 100 x 300,          │
│ +        velocity, ndarray, 100 x 300,          │
│ +        velocity_u, ndarray, 100 x 300,        │
│      obsp:                                      │
│          distances, csr_matrix, 100 x 100,      │
│          connectivities, csr_matrix, 100 x 100, │
│                                                 │
╰─────────────────────────────────────────────────╯
[11:16:02] INFO     pyrovelocity.tasks.preprocess cell state variable: clusters
╭───────────── UMAP and Leiden diff ─────────────╮
│                                                │
│ ---                                            │
│ +++                                            │
│ @@ -14,6 +14,7 @@                              │
│          initial_size_spliced, float64, 99,    │
│          initial_size, float64, 99,            │
│          n_counts, float64, 5,                 │
│ +        leiden, category, 5, [0, 1, 2, 3, 4], │
│      var:                                      │
│          true_t_, float64, 4,                  │
│          true_alpha, float64, 1,               │
│ @@ -49,8 +50,11 @@                             │
│          neighbors, dict,                      │
│          recover_dynamics, dict,               │
│          velocity_params, dict,                │
│ +        umap, dict,                           │
│ +        leiden, dict,                         │
│      obsm:                                     │
│          X_pca, ndarray, 100 x 50,             │
│ +        X_umap, ndarray, 100 x 2,             │
│      varm:                                     │
│          PCs, ndarray, 300 x 50,               │
│          loss, ndarray, 300 x 15,              │
│                                                │
╰────────────────────────────────────────────────╯
computing velocity graph (using 16/16 cores)
  0%|          | 0/100 [00:00<?, ?cells/s]
    finished (0:00:00) --> added
    'velocity_graph', sparse matrix with cosine correlations (adata.uns)
╭──────────────── Velocity graph diff ────────────────╮
│                                                     │
│ ---                                                 │
│ +++                                                 │
│ @@ -15,6 +15,7 @@                                   │
│          initial_size, float64, 99,                 │
│          n_counts, float64, 5,                      │
│          leiden, category, 5, [0, 1, 2, 3, 4],      │
│ +        velocity_self_transition, float32, 78,     │
│      var:                                           │
│          true_t_, float64, 4,                       │
│          true_alpha, float64, 1,                    │
│ @@ -52,6 +53,8 @@                                   │
│          velocity_params, dict,                     │
│          umap, dict,                                │
│          leiden, dict,                              │
│ +        velocity_graph, csr_matrix, 100 x 100,     │
│ +        velocity_graph_neg, csr_matrix, 100 x 100, │
│      obsm:                                          │
│          X_pca, ndarray, 100 x 50,                  │
│          X_umap, ndarray, 100 x 2,                  │
│                                                     │
╰─────────────────────────────────────────────────────╯
computing velocity embedding
    finished (0:00:00) --> added
    'velocity_umap', embedded velocity vectors (adata.obsm)
╭───────── Velocity embedding diff ─────────╮
│                                           │
│ ---                                       │
│ +++                                       │
│ @@ -58,6 +58,7 @@                         │
│      obsm:                                │
│          X_pca, ndarray, 100 x 50,        │
│          X_umap, ndarray, 100 x 2,        │
│ +        velocity_umap, ndarray, 100 x 2, │
│      varm:                                │
│          PCs, ndarray, 300 x 50,          │
│          loss, ndarray, 300 x 15,         │
│                                           │
╰───────────────────────────────────────────╯
computing terminal states
    identified 1 region of root cells and 1 region of end points .
    finished (0:00:00) --> added
    'root_cells', root cells of Markov diffusion process (adata.obs)
    'end_points', end points of Markov diffusion process (adata.obs)
computing latent time using root_cells as prior
    finished (0:00:00) --> added
    'latent_time', shared time (adata.obs)
╭─────────────── Latent time diff ────────────────╮
│                                                 │
│ ---                                             │
│ +++                                             │
│ @@ -16,6 +16,10 @@                              │
│          n_counts, float64, 5,                  │
│          leiden, category, 5, [0, 1, 2, 3, 4],  │
│          velocity_self_transition, float32, 78, │
│ +        root_cells, float64, 55,               │
│ +        end_points, float64, 56,               │
│ +        velocity_pseudotime, float64, 95,      │
│ +        latent_time, float64, 78,              │
│      var:                                       │
│          true_t_, float64, 4,                   │
│          true_alpha, float64, 1,                │
│                                                 │
╰─────────────────────────────────────────────────╯
╭──────────── Preprocessing summary diff ─────────────╮
│                                                     │
│ ---                                                 │
│ +++                                                 │
│ @@ -1,13 +1,83 @@                                   │
│                                                     │
│  AnnData object with n_obs × n_vars = 100 × 300     │
│      obs:                                           │
│          true_t, float64, 94,                       │
│ +        n_genes_by_counts, category, 37,           │
│ +        total_counts, float64, 99,                 │
│ +        total_counts_mt, float64, 1,               │
│ +        pct_counts_mt, float64, 1,                 │
│ +        total_counts_ribo, float64, 1,             │
│ +        pct_counts_ribo, float64, 1,               │
│ +        u_lib_size_raw, float64, 97,               │
│ +        s_lib_size_raw, float64, 99,               │
│ +        initial_size_unspliced, float64, 97,       │
│ +        initial_size_spliced, float64, 99,         │
│ +        initial_size, float64, 99,                 │
│ +        n_counts, float64, 5,                      │
│ +        leiden, category, 5, [0, 1, 2, 3, 4],      │
│ +        velocity_self_transition, float32, 78,     │
│ +        root_cells, float64, 55,                   │
│ +        end_points, float64, 56,                   │
│ +        velocity_pseudotime, float64, 95,          │
│ +        latent_time, float64, 78,                  │
│      var:                                           │
│          true_t_, float64, 4,                       │
│          true_alpha, float64, 1,                    │
│          true_beta, float64, 1,                     │
│          true_gamma, float64, 1,                    │
│          true_scaling, float64, 1,                  │
│ +        mt, bool, 1,                               │
│ +        ribo, bool, 1,                             │
│ +        n_cells_by_counts, int64, 52,              │
│ +        mean_counts, float64, 243,                 │
│ +        pct_dropout_by_counts, float64, 52,        │
│ +        total_counts, float64, 243,                │
│ +        fit_r2, float64, 300,                      │
│ +        fit_alpha, float64, 232,                   │
│ +        fit_beta, float64, 232,                    │
│ +        fit_gamma, float64, 232,                   │
│ +        fit_t_, float64, 231,                      │
│ +        fit_scaling, float64, 232,                 │
│ +        fit_std_u, float64, 232,                   │
│ +        fit_std_s, float64, 232,                   │
│ +        fit_likelihood, float64, 232,              │
│ +        fit_u0, float64, 1,                        │
│ +        fit_s0, float64, 1,                        │
│ +        fit_pval_steady, float64, 232,             │
│ +        fit_steady_u, float64, 232,                │
│ +        fit_steady_s, float64, 232,                │
│ +        fit_variance, float64, 232,                │
│ +        fit_alignment_scaling, float64, 231,       │
│ +        velocity_genes, bool, 2,                   │
│ +    uns:                                           │
│ +        log1p, dict,                               │
│ +        pca, dict,                                 │
│ +        neighbors, dict,                           │
│ +        recover_dynamics, dict,                    │
│ +        velocity_params, dict,                     │
│ +        umap, dict,                                │
│ +        leiden, dict,                              │
│ +        velocity_graph, csr_matrix, 100 x 100,     │
│ +        velocity_graph_neg, csr_matrix, 100 x 100, │
│ +    obsm:                                          │
│ +        X_pca, ndarray, 100 x 50,                  │
│ +        X_umap, ndarray, 100 x 2,                  │
│ +        velocity_umap, ndarray, 100 x 2,           │
│ +    varm:                                          │
│ +        PCs, ndarray, 300 x 50,                    │
│ +        loss, ndarray, 300 x 15,                   │
│      layers:                                        │
│          spliced, ndarray, 100 x 300,               │
│          unspliced, ndarray, 100 x 300,             │
│ +        raw_unspliced, csr_matrix, 100 x 300,      │
│ +        raw_spliced, csr_matrix, 100 x 300,        │
│ +        Ms, ndarray, 100 x 300,                    │
│ +        Mu, ndarray, 100 x 300,                    │
│ +        fit_t, ndarray, 100 x 300,                 │
│ +        fit_tau, ndarray, 100 x 300,               │
│ +        fit_tau_, ndarray, 100 x 300,              │
│ +        velocity, ndarray, 100 x 300,              │
│ +        velocity_u, ndarray, 100 x 300,            │
│ +    obsp:                                          │
│ +        distances, csr_matrix, 100 x 100,          │
│ +        connectivities, csr_matrix, 100 x 100,     │
│                                                     │
╰─────────────────────────────────────────────────────╯
[11:16:03] INFO     pyrovelocity.tasks.preprocess successfully generated data/processed/simulated_processed.h5ad
```
