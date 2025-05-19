# Example parquet directory tree from AnnData object

```sh
❯ tree --du -ah postprocessed_pancreas_50_7.pqdata
[277K]  postprocessed_pancreas_50_7.pqdata
├── [ 53K]  layers
│   ├── [5.0K]  fit_t.parquet
│   ├── [4.0K]  fit_tau_.parquet
│   ├── [4.1K]  fit_tau.parquet
│   ├── [4.0K]  Ms.parquet
│   ├── [4.1K]  Mu.parquet
│   ├── [2.7K]  raw_spliced.parquet
│   ├── [2.8K]  raw_unspliced.parquet
│   ├── [4.5K]  spliced_pyro.parquet
│   ├── [3.3K]  spliced.parquet
│   ├── [3.5K]  unspliced.parquet
│   ├── [4.5K]  velocity_pyro.parquet
│   ├── [5.0K]  velocity_u.parquet
│   └── [5.0K]  velocity.parquet
├── [ 28K]  obs.parquet
├── [ 14K]  obsm
│   ├── [4.6K]  velocity_pyro_pca.parquet
│   ├── [1.8K]  velocity_pyro_umap.parquet
│   ├── [1.8K]  velocity_umap.parquet
│   ├── [4.0K]  X_pca.parquet
│   └── [1.7K]  X_umap.parquet
├── [ 58K]  obsp
│   ├── [ 28K]  connectivities.parquet
│   └── [ 30K]  distances.parquet
├── [ 95K]  uns
│   ├── [ 789]  clusters_coarse_colors.parquet
│   ├── [ 780]  clusters_colors.parquet
│   ├── [ 700]  day_colors.parquet
│   ├── [1.7K]  pca
│   │   ├── [ 808]  variance_ratio.parquet
│   │   └── [ 784]  variance.parquet
│   ├── [ 21K]  velocity_graph_neg.parquet
│   ├── [ 22K]  velocity_graph.parquet
│   ├── [1.5K]  velocity_params
│   │   ├── [ 685]  embeddings.parquet
│   │   └── [ 741]  perc.parquet
│   ├── [ 28K]  velocity_pyro_graph_neg.parquet
│   ├── [ 17K]  velocity_pyro_graph.parquet
│   └── [ 786]  velocity_pyro_params
│       └── [ 690]  embeddings.parquet
├── [ 750]  uns.json
├── [ 17K]  var.parquet
├── [7.2K]  varm
│   ├── [4.7K]  loss.parquet
│   └── [2.4K]  PCs.parquet
└── [3.2K]  X.parquet

 277K used in 9 directories, 38 files
 ```
