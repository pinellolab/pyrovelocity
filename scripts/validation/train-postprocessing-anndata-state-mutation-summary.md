# Train postprocess AnnData state mutation summary

```python
[ins] In [1]:         >>> # xdoctest: +SKIP
         ...:         >>> from pyrovelocity.tasks.postprocess import postprocess_dataset
         ...:         >>> from pyrovelocity.tasks.train import train_dataset
         ...:         >>> from pyrovelocity.utils import generate_sample_data, print_anndata, pretty_print_dict
         ...:         >>> from pyrovelocity.tasks.preprocess import copy_raw_counts
         ...:         >>> from pyrovelocity.io import CompressedPickle
         ...:         >>> from pathlib import Path
         ...:         >>> import scanpy as sc
         ...:         >>> tmpdir = None
         ...:         >>> try:
         ...:         >>>     tmp = getfixture("tmp_path")
         ...:         >>> except NameError:
         ...:         >>>     import tempfile
         ...:         >>>     tmpdir = tempfile.TemporaryDirectory()
         ...:         >>>     tmp = Path(tmpdir.name)
         ...:         >>> adata = generate_sample_data(random_seed=99)
         ...:         >>> copy_raw_counts(adata)
         ...:         >>>
         ...:         >>> # Compute embeddings needed for velocity visualization
         ...:         >>> sc.pp.pca(adata, random_state=99)
         ...:         >>> sc.pp.neighbors(adata, n_neighbors=10, random_state=99)
         ...:         >>> sc.tl.umap(adata, random_state=99)
         ...:         >>> sc.tl.leiden(adata, random_state=99)
         ...:         >>>
         ...:         >>> # Train the model and get all the paths
         ...:         >>> data_model, data_model_path, trained_data_path, model_path, posterior_samples_path, metrics_path, _, _, _ = train_dataset(
         ...:         ...   adata,
         ...:         ...   data_set_name="simulated",
         ...:         ...   model_identifier="model2",
         ...:         ...   models_path=tmp / "models",
         ...:         ...   use_gpu="auto",
         ...:         ...   random_seed=99,
         ...:         ...   max_epochs=200,
         ...:         ...   force=True,
         ...:         ... )
         ...:         >>> # Use the paths returned by train_dataset for postprocessing
         ...:         >>> pyrovelocity_data_path, postprocessed_data_path = postprocess_dataset(
         ...:         ...     data_model=data_model,
         ...:         ...     data_model_path=data_model_path,
         ...:         ...     trained_data_path=trained_data_path,
         ...:         ...     model_path=model_path,
         ...:         ...     posterior_samples_path=posterior_samples_path,
         ...:         ...     metrics_path=metrics_path,
         ...:         ...     vector_field_basis="umap",  # Use umap for vector field visualization
         ...:         ...     number_posterior_samples=3,
         ...:         ...     random_seed=99,
         ...:         ... )
         ...:         >>> adata = sc.read_h5ad(postprocessed_data_path)
         ...:         >>> posterior_samples = CompressedPickle.load(posterior_samples_path)
         ...:         >>> pyrovelocity_data = CompressedPickle.load(pyrovelocity_data_path)
         ...:         >>> print_anndata(adata)
         ...:         >>> pretty_print_dict(posterior_samples)
         ...:         >>> pretty_print_dict(pyrovelocity_data)
         ...:         >>> # Handle temporary directory cleanup
         ...:         >>> keep_tmp = True
         ...:         >>> if tmpdir is not None:
         ...:         >>>     if keep_tmp:
         ...:         >>>         tmp_dir_path = tmpdir.name
         ...:         >>>         tmpdir._finalizer.detach()
         ...:         >>>         print(f"\nTemporary directory preserved at: {tmp_dir_path}")
         ...:         >>>     else:
         ...:         >>>         print("\nTemporary directory will be cleaned up on exit.")
         ...:
[14:33:15] INFO     pyrovelocity.tasks.preprocess 'raw_unspliced' key added raw unspliced counts to adata.layers
           INFO     pyrovelocity.tasks.preprocess 'raw_spliced' key added raw spliced counts added to adata.layers
           INFO     pyrovelocity.tasks.preprocess 'u_lib_size_raw' key added unspliced library size to adata.obs, total: 5503.0
           INFO     pyrovelocity.tasks.preprocess 's_lib_size_raw' key added spliced library size to adata.obs, total: 5550.0
[14:33:17] INFO     pyrovelocity.tasks.train Reset random state from seed: 99
           INFO     pyrovelocity.tasks.train

                    Training: simulated_model2


           INFO     pyrovelocity.tasks.train

                    Verifying existence of paths for:

                      model data: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2

           INFO     pyrovelocity.tasks.train Accelerator type specified as auto resolves to:
                            accelerator: cpu
                            devices: 1
                            device: cpu

           INFO     pyrovelocity.tasks.train Training model: simulated_model2
Active run_id: 1fb538b5b4e947449a91d5a538f19a17
           INFO     pyrovelocity.tasks.train Reset random state from seed: 99
           INFO     pyrovelocity.tasks.train AnnData object prior to model training

AnnData object with n_obs × n_vars = 100 × 12
    obs:
        true_t, float64, 94,
        u_lib_size_raw, float64, 46,
        s_lib_size_raw, float64, 52,
        leiden, category, 8, [0, 1, 2, 3, 4, 5, 6, 7],
    var:
        true_t_, float64, 4,
        true_alpha, float64, 1,
        true_beta, float64, 1,
        true_gamma, float64, 1,
        true_scaling, float64, 1,
    uns:
        pca, dict,
        neighbors, dict,
        umap, dict,
        leiden, dict,
    obsm:
        X_pca, ndarray, 100 x 11,
        X_umap, ndarray, 100 x 2,
    varm:
        PCs, ndarray, 12 x 11,
    layers:
        unspliced, ndarray, 100 x 12,
        spliced, ndarray, 100 x 12,
        raw_unspliced, ndarray, 100 x 12,
        raw_spliced, ndarray, 100 x 12,
    obsp:
        distances, csr_matrix, 100 x 100,
        connectivities, csr_matrix, 100 x 100,
╭──────────── PyroVelocity.setup_anndata diff ────────────╮
│                                                         │
│ ---                                                     │
│ +++                                                     │
│ @@ -5,6 +5,13 @@                                        │
│          u_lib_size_raw, float64, 46,                   │
│          s_lib_size_raw, float64, 52,                   │
│          leiden, category, 8, [0, 1, 2, 3, 4, 5, 6, 7], │
│ +        u_lib_size, float64, 46,                       │
│ +        s_lib_size, float64, 52,                       │
│ +        u_lib_size_mean, float64, 1,                   │
│ +        s_lib_size_mean, float64, 1,                   │
│ +        u_lib_size_scale, float64, 1,                  │
│ +        s_lib_size_scale, float64, 1,                  │
│ +        ind_x, int64, 100,                             │
│      var:                                               │
│          true_t_, float64, 4,                           │
│          true_alpha, float64, 1,                        │
│ @@ -16,6 +23,8 @@                                       │
│          neighbors, dict,                               │
│          umap, dict,                                    │
│          leiden, dict,                                  │
│ +        _scvi_uuid, str,                               │
│ +        _scvi_manager_uuid, str,                       │
│      obsm:                                              │
│          X_pca, ndarray, 100 x 11,                      │
│          X_umap, ndarray, 100 x 2,                      │
│                                                         │
╰─────────────────────────────────────────────────────────╯
           INFO     pyrovelocity.models._velocity n_U: 12
                    n_cells: 100
                    n_vars: 12

           INFO     pyrovelocity.models._velocity_module Model type: auto, Guide type: auto
           INFO     pyrovelocity.models._velocity_model Initializing LogNormalModel
           INFO     pyrovelocity.models._velocity Model initialized
           INFO     pyrovelocity.models._trainer train model in a single batch with all data loaded into accelerator memory
           INFO     pyrovelocity.models._trainer
                    Loading model with:
                            accelerator: cpu
                            devices: 1
                            device: cpu


           INFO     pyrovelocity.models._trainer step    1 loss = 20.6598 patience = 45
[14:33:18] INFO     pyrovelocity.models._trainer step  100 loss = 16.1126 patience = 45
           INFO     pyrovelocity.models._trainer step  200 loss = 15.0122 patience = 45
╭─ Posterior samples generation diff ─╮
│                                     │
│                                     │
│                                     │
╰─────────────────────────────────────╯
[14:33:19] INFO     pyrovelocity.tasks.train AnnData object after model training
           INFO     pyrovelocity.utils
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
                        var:
                            true_t_, float64, 4,
                            true_alpha, float64, 1,
                            true_beta, float64, 1,
                            true_gamma, float64, 1,
                            true_scaling, float64, 1,
                        uns:
                            pca, dict,
                            neighbors, dict,
                            umap, dict,
                            leiden, dict,
                            _scvi_uuid, str,
                            _scvi_manager_uuid, str,
                        obsm:
                            X_pca, ndarray, 100 x 11,
                            X_umap, ndarray, 100 x 2,
                        varm:
                            PCs, ndarray, 12 x 11,
                        layers:
                            unspliced, ndarray, 100 x 12,
                            spliced, ndarray, 100 x 12,
                            raw_unspliced, ndarray, 100 x 12,
                            raw_spliced, ndarray, 100 x 12,
                        obsp:
                            distances, csr_matrix, 100 x 100,
                            connectivities, csr_matrix, 100 x 100,
           INFO     pyrovelocity.tasks.train Data attributes after model training
           INFO     pyrovelocity.tasks.train Saving posterior samples: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/posterior_samples.pkl.zst

Array 'u':
  Shape: (30, 100, 12)
  Density: 16.05%
  Original size: 144000 bytes
  Sparse size: 161784 bytes
  Size reduction: -0.12349999999999994 bytes

Array 's':
  Shape: (30, 100, 12)
  Density: 16.09%
  Original size: 144000 bytes
  Sparse size: 162232 bytes
  Size reduction: -0.126611111111111 bytes

           INFO     pyrovelocity.io.compressedpickle
                    Successfully saved file: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/posterior_samples.pkl.zst
                    SHA-256 hash: 38955ab7ab14c0db56c138f3495b5f29e1f09195c25fc0cdcdf949efd8fbc3ce

           INFO     pyrovelocity.tasks.train
                    run_id: 1fb538b5b4e947449a91d5a538f19a17
                    artifacts: []
                    params: {'model_identifier': 'model2', 'data_set_name': 'simulated', 'model_type': 'auto', 'guide_type': 'auto', 'batch_size': '-1'}
                    metrics: {'real_epochs': 200.0, '-ELBO': -15.012196792090933}
                    tags: {}

           INFO     pyrovelocity.utils
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
                        var:
                            true_t_, float64, 4,
                            true_alpha, float64, 1,
                            true_beta, float64, 1,
                            true_gamma, float64, 1,
                            true_scaling, float64, 1,
                        uns:
                            pca, dict,
                            neighbors, dict,
                            umap, dict,
                            leiden, dict,
                            _scvi_uuid, str,
                            _scvi_manager_uuid, str,
                        obsm:
                            X_pca, ndarray, 100 x 11,
                            X_umap, ndarray, 100 x 2,
                        varm:
                            PCs, ndarray, 12 x 11,
                        layers:
                            unspliced, ndarray, 100 x 12,
                            spliced, ndarray, 100 x 12,
                            raw_unspliced, ndarray, 100 x 12,
                            raw_spliced, ndarray, 100 x 12,
                        obsp:
                            distances, csr_matrix, 100 x 100,
                            connectivities, csr_matrix, 100 x 100,
           INFO     pyrovelocity.tasks.train Saving trained data: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/trained.h5ad
           INFO     pyrovelocity.tasks.train Saving model: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/model
           INFO     pyrovelocity.tasks.train
                    Returning paths to saved data:

                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2
                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/trained.h5ad
                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/model
                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/posterior_samples.pkl.zst
                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/metrics.json
                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/run_info.json
                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/ELBO.png
                    /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/ELBO.csv

           INFO     pyrovelocity.tasks.postprocess Reset random state from seed: 99
           INFO     pyrovelocity.tasks.postprocess using 13 cpus
           INFO     pyrovelocity.tasks.postprocess Loading trained data: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/trained.h5ad

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
    var:
        true_t_, float64, 4,
        true_alpha, float64, 1,
        true_beta, float64, 1,
        true_gamma, float64, 1,
        true_scaling, float64, 1,
    uns:
        _scvi_manager_uuid, str,
        _scvi_uuid, str,
        leiden, dict,
        neighbors, dict,
        pca, dict,
        umap, dict,
    obsm:
        X_pca, ndarray, 100 x 11,
        X_umap, ndarray, 100 x 2,
    varm:
        PCs, ndarray, 12 x 11,
    layers:
        raw_spliced, ndarray, 100 x 12,
        raw_unspliced, ndarray, 100 x 12,
        spliced, ndarray, 100 x 12,
        unspliced, ndarray, 100 x 12,
    obsp:
        connectivities, csr_matrix, 100 x 100,
        distances, csr_matrix, 100 x 100,
           INFO     pyrovelocity.tasks.postprocess Loading posterior samples: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/posterior_samples.pkl.zst
           INFO     pyrovelocity.io.compressedpickle
                    Successfully loaded file: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/posterior_samples.pkl.zst
                    SHA-256 hash: 38955ab7ab14c0db56c138f3495b5f29e1f09195c25fc0cdcdf949efd8fbc3ce

           INFO     pyrovelocity.tasks.postprocess Using 3 posterior samples from 30 posterior samples
           INFO     pyrovelocity.utils
                    alpha:
                    <class 'tuple'>
                    (3, 1, 12)

                    gamma:
                    <class 'tuple'>
                    (3, 1, 12)

                    beta:
                    <class 'tuple'>
                    (3, 1, 12)

                    u_offset:
                    <class 'tuple'>
                    (3, 1, 12)

                    s_offset:
                    <class 'tuple'>
                    (3, 1, 12)

                    t0:
                    <class 'tuple'>
                    (3, 1, 12)

                    u_scale:
                    <class 'tuple'>
                    (3, 1, 12)

                    dt_switching:
                    <class 'tuple'>
                    (3, 1, 12)

                    u_inf:
                    <class 'tuple'>
                    (3, 1, 12)

                    s_inf:
                    <class 'tuple'>
                    (3, 1, 12)

                    switching:
                    <class 'tuple'>
                    (3, 1, 12)

                    cell_time:
                    <class 'tuple'>
                    (3, 100, 1)

                    u_read_depth:
                    <class 'tuple'>
                    (3, 100, 1)

                    s_read_depth:
                    <class 'tuple'>
                    (3, 100, 1)

                    cell_gene_state:
                    <class 'tuple'>
                    (3, 100, 12)

                    ut:
                    <class 'tuple'>
                    (3, 100, 12)

                    st:
                    <class 'tuple'>
                    (3, 100, 12)

                    u:
                    <class 'tuple'>
                    (3, 100, 12)

                    s:
                    <class 'tuple'>
                    (3, 100, 12)


           INFO     pyrovelocity.tasks.postprocess Loading model data: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/model
           INFO     pyrovelocity.models._velocity
                    Loading model with:
                            accelerator: cpu
                            devices: 1
                            device: cpu

INFO     File /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/model/model.pt already downloaded
           INFO     pyrovelocity.models._velocity n_U: 12
                    n_cells: 100
                    n_vars: 12

           INFO     pyrovelocity.models._velocity_module Model type: auto, Guide type: auto
           INFO     pyrovelocity.models._velocity_model Initializing LogNormalModel
           INFO     pyrovelocity.models._velocity Model initialized
           INFO     pyrovelocity.models._velocity Preparing underlying `PyroBaseModuleClass` module for load
           INFO     pyrovelocity.models._trainer
                    train model:
                            training fraction: 0.995
                            validation fraction: 0.005

GPU available: True (mps), used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
           INFO     pyrovelocity.models._trainer
                    train model:
                            training fraction: 0.8
                            validation fraction: 0.2

GPU available: True (mps), used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Epoch 1/1:   0%|                                                                                                                                                        | 0/1 [00:00<?, ?it/s]           INFO     root Guessed max_plate_nesting = 2
Epoch 1/1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 30.56it/s, v_num=1]`Trainer.fit` stopped: `max_epochs=1` reached.
Epoch 1/1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 26.83it/s, v_num=1]
           INFO     pyrovelocity.tasks.postprocess Postprocessing model data: simulated_model2
Active run_id: 491180da61bf455d819193b647789852
           INFO     pyrovelocity.utils
                    alpha:
                    <class 'tuple'>
                    (3, 1, 12)

                    gamma:
                    <class 'tuple'>
                    (3, 1, 12)

                    beta:
                    <class 'tuple'>
                    (3, 1, 12)

                    u_offset:
                    <class 'tuple'>
                    (3, 1, 12)

                    s_offset:
                    <class 'tuple'>
                    (3, 1, 12)

                    t0:
                    <class 'tuple'>
                    (3, 1, 12)

                    u_scale:
                    <class 'tuple'>
                    (3, 1, 12)

                    dt_switching:
                    <class 'tuple'>
                    (3, 1, 12)

                    u_inf:
                    <class 'tuple'>
                    (3, 1, 12)

                    s_inf:
                    <class 'tuple'>
                    (3, 1, 12)

                    switching:
                    <class 'tuple'>
                    (3, 1, 12)

                    cell_time:
                    <class 'tuple'>
                    (3, 100, 1)

                    u_read_depth:
                    <class 'tuple'>
                    (3, 100, 1)

                    s_read_depth:
                    <class 'tuple'>
                    (3, 100, 1)

                    cell_gene_state:
                    <class 'tuple'>
                    (3, 100, 12)

                    ut:
                    <class 'tuple'>
                    (3, 100, 12)

                    st:
                    <class 'tuple'>
                    (3, 100, 12)

                    u:
                    <class 'tuple'>
                    (3, 100, 12)

                    s:
                    <class 'tuple'>
                    (3, 100, 12)


           INFO     pyrovelocity.tasks.postprocess Computing vector field uncertainty
           INFO     pyrovelocity.analysis.analyze Estimating vector field uncertainty
computing velocity graph (using 13/16 cores)
  0%|          | 0/100 [00:00<?, ?cells/s]
    finished (0:00:03) --> added
    'velocity_pyro_graph', sparse matrix with cosine correlations (adata.uns)
computing velocity embedding
    finished (0:00:00) --> added
    'velocity_pyro_umap', embedded velocity vectors (adata.obsm)
computing velocity graph (using 13/16 cores)
  0%|          | 0/100 [00:00<?, ?cells/s]
    finished (0:00:00) --> added
    'velocity_pyro_graph', sparse matrix with cosine correlations (adata.uns)
computing velocity embedding
    finished (0:00:00) --> added
    'velocity_pyro_umap', embedded velocity vectors (adata.obsm)
computing velocity graph (using 13/16 cores)
  0%|          | 0/100 [00:00<?, ?cells/s]
    finished (0:00:00) --> added
    'velocity_pyro_graph', sparse matrix with cosine correlations (adata.uns)
computing velocity embedding
    finished (0:00:00) --> added
    'velocity_pyro_umap', embedded velocity vectors (adata.obsm)
[14:33:23] INFO     pyrovelocity.analysis.analyze Computing mean vector field
computing velocity graph (using 13/16 cores)
  0%|          | 0/100 [00:00<?, ?cells/s]
    finished (0:00:00) --> added
    'velocity_pyro_graph', sparse matrix with cosine correlations (adata.uns)
computing velocity embedding
    finished (0:00:00) --> added
    'velocity_pyro_umap', embedded velocity vectors (adata.obsm)
           INFO     pyrovelocity.analysis.analyze Estimating vector field uncertainty
WARNING: Directly projecting velocities into PCA space is for exploratory analysis on principal components.
         It does not reflect the actual velocity field from high dimensional gene expression space.
         To visualize velocities, consider applying `direct_pca_projection=False`.

computing velocity embedding
    finished (0:00:00) --> added
    'velocity_pyro_pca', embedded velocity vectors (adata.obsm)
WARNING: Directly projecting velocities into PCA space is for exploratory analysis on principal components.
         It does not reflect the actual velocity field from high dimensional gene expression space.
         To visualize velocities, consider applying `direct_pca_projection=False`.

computing velocity embedding
    finished (0:00:00) --> added
    'velocity_pyro_pca', embedded velocity vectors (adata.obsm)
WARNING: Directly projecting velocities into PCA space is for exploratory analysis on principal components.
         It does not reflect the actual velocity field from high dimensional gene expression space.
         To visualize velocities, consider applying `direct_pca_projection=False`.

computing velocity embedding
    finished (0:00:00) --> added
    'velocity_pyro_pca', embedded velocity vectors (adata.obsm)
╭─────── Vector field uncertainty computation diff ────────╮
│                                                          │
│ ---                                                      │
│ +++                                                      │
│ @@ -12,12 +12,14 @@                                      │
│          u_lib_size_scale, float64, 1,                   │
│          s_lib_size_scale, float64, 1,                   │
│          ind_x, int64, 100,                              │
│ +        velocity_pyro_self_transition, float32, 99,     │
│      var:                                                │
│          true_t_, float64, 4,                            │
│          true_alpha, float64, 1,                         │
│          true_beta, float64, 1,                          │
│          true_gamma, float64, 1,                         │
│          true_scaling, float64, 1,                       │
│ +        velocity_genes, bool, 1,                        │
│      uns:                                                │
│          _scvi_manager_uuid, str,                        │
│          _scvi_uuid, str,                                │
│ @@ -25,9 +27,14 @@                                       │
│          neighbors, dict,                                │
│          pca, dict,                                      │
│          umap, dict,                                     │
│ +        velocity_pyro_graph, csr_matrix, 100 x 100,     │
│ +        velocity_pyro_graph_neg, csr_matrix, 100 x 100, │
│ +        velocity_pyro_params, dict,                     │
│      obsm:                                               │
│          X_pca, ndarray, 100 x 11,                       │
│          X_umap, ndarray, 100 x 2,                       │
│ +        velocity_pyro_umap, ndarray, 100 x 2,           │
│ +        velocity_pyro_pca, ndarray, 100 x 11,           │
│      varm:                                               │
│          PCs, ndarray, 12 x 11,                          │
│      layers:                                             │
│ @@ -35,6 +42,8 @@                                        │
│          raw_unspliced, ndarray, 100 x 12,               │
│          spliced, ndarray, 100 x 12,                     │
│          unspliced, ndarray, 100 x 12,                   │
│ +        spliced_pyro, ndarray, 100 x 12,                │
│ +        velocity_pyro, ndarray, 100 x 12,               │
│      obsp:                                               │
│          connectivities, csr_matrix, 100 x 100,          │
│          distances, csr_matrix, 100 x 100,               │
│                                                          │
╰──────────────────────────────────────────────────────────╯
           INFO     pyrovelocity.tasks.postprocess Data attributes after computation of vector field uncertainty
           INFO     pyrovelocity.utils
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
                            velocity_pyro_umap, ndarray, 100 x 2,
                            velocity_pyro_pca, ndarray, 100 x 11,
                        varm:
                            PCs, ndarray, 12 x 11,
                        layers:
                            raw_spliced, ndarray, 100 x 12,
                            raw_unspliced, ndarray, 100 x 12,
                            spliced, ndarray, 100 x 12,
                            unspliced, ndarray, 100 x 12,
                            spliced_pyro, ndarray, 100 x 12,
                            velocity_pyro, ndarray, 100 x 12,
                        obsp:
                            connectivities, csr_matrix, 100 x 100,
                            distances, csr_matrix, 100 x 100,
           INFO     pyrovelocity.tasks.postprocess Saving pyrovelocity data: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/pyrovelocity.pkl.zst
           INFO     pyrovelocity.io.compressedpickle
                    Successfully saved file: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/pyrovelocity.pkl.zst
                    SHA-256 hash: fc67c2c70b4eaa7cfb8cd157d19b31802e71786e1d2c84fb6b772119036364b1

╭────────────── Postprocessing summary diff ───────────────╮
│                                                          │
│ ---                                                      │
│ +++                                                      │
│ @@ -10,31 +10,40 @@                                      │
│          u_lib_size_mean, float64, 1,                    │
│          s_lib_size_mean, float64, 1,                    │
│          u_lib_size_scale, float64, 1,                   │
│          s_lib_size_scale, float64, 1,                   │
│          ind_x, int64, 100,                              │
│ +        velocity_pyro_self_transition, float32, 99,     │
│      var:                                                │
│          true_t_, float64, 4,                            │
│          true_alpha, float64, 1,                         │
│          true_beta, float64, 1,                          │
│          true_gamma, float64, 1,                         │
│          true_scaling, float64, 1,                       │
│ +        velocity_genes, bool, 1,                        │
│      uns:                                                │
│          _scvi_manager_uuid, str,                        │
│          _scvi_uuid, str,                                │
│          leiden, dict,                                   │
│          neighbors, dict,                                │
│          pca, dict,                                      │
│          umap, dict,                                     │
│ +        velocity_pyro_graph, csr_matrix, 100 x 100,     │
│ +        velocity_pyro_graph_neg, csr_matrix, 100 x 100, │
│ +        velocity_pyro_params, dict,                     │
│      obsm:                                               │
│          X_pca, ndarray, 100 x 11,                       │
│          X_umap, ndarray, 100 x 2,                       │
│ +        velocity_pyro_umap, ndarray, 100 x 2,           │
│ +        velocity_pyro_pca, ndarray, 100 x 11,           │
│      varm:                                               │
│          PCs, ndarray, 12 x 11,                          │
│      layers:                                             │
│          raw_spliced, ndarray, 100 x 12,                 │
│          raw_unspliced, ndarray, 100 x 12,               │
│          spliced, ndarray, 100 x 12,                     │
│          unspliced, ndarray, 100 x 12,                   │
│ +        spliced_pyro, ndarray, 100 x 12,                │
│ +        velocity_pyro, ndarray, 100 x 12,               │
│      obsp:                                               │
│          connectivities, csr_matrix, 100 x 100,          │
│          distances, csr_matrix, 100 x 100,               │
│                                                          │
╰──────────────────────────────────────────────────────────╯
[14:33:24] INFO     pyrovelocity.tasks.postprocess Saving postprocessed data: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/postprocessed.h5ad
           INFO     pyrovelocity.tasks.postprocess
                    run_id: 491180da61bf455d819193b647789852,
                    artifacts: [],
                    params: {},
                    metrics: {'FDR_HMP': 0.04885577959958058, 'MAE': 4.463472222222222, 'FDR_sig_frac': 0.88},
                    tags: {}


           INFO     pyrovelocity.io.compressedpickle
                    Successfully loaded file: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/posterior_samples.pkl.zst
                    SHA-256 hash: 38955ab7ab14c0db56c138f3495b5f29e1f09195c25fc0cdcdf949efd8fbc3ce

           INFO     pyrovelocity.io.compressedpickle
                    Successfully loaded file: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km/models/simulated_model2/pyrovelocity.pkl.zst
                    SHA-256 hash: fc67c2c70b4eaa7cfb8cd157d19b31802e71786e1d2c84fb6b772119036364b1

           INFO     pyrovelocity.utils
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
           INFO     pyrovelocity.utils
                    alpha:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    gamma:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    beta:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    u_offset:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    s_offset:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    t0:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    u_scale:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    dt_switching:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    u_inf:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    s_inf:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    switching:
                    <class 'numpy.ndarray'>
                    (30, 1, 12) float32

                    cell_time:
                    <class 'numpy.ndarray'>
                    (30, 100, 1) float32

                    u_read_depth:
                    <class 'numpy.ndarray'>
                    (30, 100, 1) float32

                    s_read_depth:
                    <class 'numpy.ndarray'>
                    (30, 100, 1) float32

                    cell_gene_state:
                    <class 'numpy.ndarray'>
                    (30, 100, 12) float32

                    ut:
                    <class 'numpy.ndarray'>
                    (30, 100, 12) float32

                    st:
                    <class 'numpy.ndarray'>
                    (30, 100, 12) float32

                    u:
                    <class 'numpy.ndarray'>
                    (30, 100, 12) float32

                    s:
                    <class 'numpy.ndarray'>
                    (30, 100, 12) float32


           INFO     pyrovelocity.utils
                    alpha:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    gamma:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    beta:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    u_offset:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    s_offset:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    t0:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    u_scale:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    dt_switching:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    u_inf:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    s_inf:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    switching:
                    <class 'numpy.ndarray'>
                    (3, 1, 12) float32

                    cell_time:
                    <class 'numpy.ndarray'>
                    (3, 100, 1) float32

                    u_read_depth:
                    <class 'numpy.ndarray'>
                    (3, 100, 1) float32

                    s_read_depth:
                    <class 'numpy.ndarray'>
                    (3, 100, 1) float32

                    cell_gene_state:
                    <class 'numpy.ndarray'>
                    (3, 100, 12) float32

                    gene_ranking:
                    <class 'pandas.core.frame.DataFrame'>
                           mean_mae  time_correlation  mean_mae_rank  time_correlation_rank  rank_product  selected genes
                    genes
                    5     -0.967535          0.827009            5.0                    4.0          20.0               1
                    1     -0.968357          0.749202            6.0                    6.0          36.0               1
                    2     -0.957920          0.745092            2.0                    8.0          16.0               1
                    6     -0.953181          0.708051            1.0                   10.0          10.0               1
                    10    -0.965597          0.652101            3.0                   11.0          33.0               0
                    9     -0.965629         -0.214246            4.0                   12.0          48.0               0
                    7     -1.025974         -0.723620           11.0                    9.0          99.0               0
                    3     -1.033333         -0.745621           12.0                    7.0          84.0               0
                    11    -1.018519         -0.815981           10.0                    5.0          50.0               0
                    4     -0.981818         -0.860995            9.0                    3.0          27.0               0
                    8     -0.979703         -0.939102            8.0                    2.0          16.0               0
                    0     -0.978519         -0.941670            7.0                    1.0           7.0               0

                    original_spaces_embeds_magnitude:
                    <class 'numpy.ndarray'>
                    (3, 100) float32

                    genes:
                    <class 'list'>
                    ['5', '1', '2', '6']

                    vector_field_posterior_samples:
                    <class 'numpy.ndarray'>
                    (3, 100, 2) float64

                    vector_field_posterior_mean:
                    <class 'numpy.ndarray'>
                    (100, 2) float64

                    fdri:
                    <class 'numpy.ndarray'>
                    (100,) float64

                    embeds_magnitude:
                    <class 'numpy.ndarray'>
                    (3, 100) float64

                    embeds_angle:
                    <class 'numpy.ndarray'>
                    (3, 100) float64

                    ut_mean:
                    <class 'numpy.ndarray'>
                    (100, 12) float32

                    st_mean:
                    <class 'numpy.ndarray'>
                    (100, 12) float32

                    ut_std:
                    <class 'numpy.ndarray'>
                    (100, 12) float32

                    st_std:
                    <class 'numpy.ndarray'>
                    (100, 12) float32

                    pca_vector_field_posterior_samples:
                    <class 'numpy.ndarray'>
                    (3, 100, 11) float64

                    pca_embeds_angle:
                    <class 'numpy.ndarray'>
                    (3, 100) float64

                    pca_fdri:
                    <class 'numpy.ndarray'>
                    (100,) float64



Temporary directory preserved at: /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km
[ins] In [2]: adata.uns["velocity_pyro_params"]
Out[2]:
{'embeddings': array(['umap', 'pca', 'pca', 'pca'], dtype=object),
 'mode_neighbors': 'distances',
 'n_recurse_neighbors': 2}
```

The associated artifacts are

```sh
❯ tree --du -ah /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km
[855K]  /var/folders/gm/sczlkg_x4kd39019wn_f7hy80000gn/T/tmpvb_fm5km
└── [855K]  models
    └── [855K]  simulated_model2
        ├── [4.5K]  ELBO.csv
        ├── [9.5K]  ELBO.png
        ├── [ 154]  metrics.json
        ├── [ 23K]  model
        │   ├── [ 13K]  model.pt
        │   └── [ 10K]  param_store_test.pt
        ├── [330K]  posterior_samples.pkl.zst
        ├── [278K]  postprocessed.h5ad
        ├── [ 64K]  pyrovelocity.pkl.zst
        ├── [ 447]  run_info.json
        └── [145K]  trained.h5ad

 855K used in 4 directories, 10 files
```
