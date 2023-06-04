import os

import celltypist
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import scanpy as sc
from pyensembl import EnsemblRelease


DATA_FILE_PATH = "data/external/pbmc10k.h5ad"
BACKUP_URL = "https://storage.googleapis.com/pyrovelocity/data/pbmc10k.h5ad"
ENSEMBL_RELEASE = 98
# See model descriptions at https://www.celltypist.org/models
# Low here refers to inclusion of cell types at the
# leaves of the hematopoietic tree.
# High refers to cutting the tree closer to its root.
# We rename these to HIGH_RESOLUTION_MODEL and LOW_RESOLUTION_MODEL
# respectively, though it is impossible to avoid potential confusion.
HIGH_RESOLUTION_MODEL = "Immune_All_Low.pkl"
LOW_RESOLUTION_MODEL = "Immune_All_High.pkl"
LOW_RES_PLOT_FILENAME = "labeled_pbmc10k_low_resolution.pdf"
HIGH_RES_PLOT_FILENAME = "labeled_pbmc10k_high_resolution.pdf"


def load_data(file_path, backup_url):
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Downloading from backup URL.")
        file_path = backup_url
    return sc.read(file_path)


def rename_var_to_gene_symbol(adata, release):
    ensembl = EnsemblRelease(release)
    gene_ids = adata.var_names.tolist()
    gene_symbols = [
        ensembl.gene_name_of_gene_id(gene_id)
        if ensembl.gene_by_id(gene_id)
        else gene_id
        for gene_id in gene_ids
    ]
    adata.var_names = gene_symbols
    adata = adata[:, ~adata.var_names.isin([""])]
    return adata


def preprocess_data(adata, target_sum=1e4):
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=target_sum)
    sc.pp.log1p(adata_copy)
    adata.raw = adata_copy
    return adata


def label_cell_types(adata, model, obs_key):
    model = celltypist.models.Model.load(model)
    predictions = celltypist.annotate(adata, model=model, majority_voting=True)
    adata.obs[obs_key] = predictions.predicted_labels["majority_voting"]
    return adata


def plot_umap(adata, plot_filename, obs_key):
    cmap = matplotlib.colormaps["tab20"]
    classes = adata.obs[obs_key].cat.categories
    colors = [cmap(i) for i in range(len(classes))]
    adata.uns[f"{obs_key}_colors"] = colors

    sc.pl.umap(
        adata,
        color=obs_key,
        show=False,
        legend_loc=None,
        frameon=False,
        title="",
    )

    ax = plt.gca()
    ax.axis("off")

    lines = [
        mlines.Line2D(
            [], [], color=colors[i], marker="o", markersize=10, linestyle="None"
        )
        for i in range(len(classes))
    ]
    legend_labels = classes.tolist()

    legend = ax.legend(
        handles=lines,
        labels=legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )

    plt.savefig(plot_filename, bbox_extra_artists=(legend,), bbox_inches="tight")

    plt.show()


def main():
    adata = load_data(DATA_FILE_PATH, BACKUP_URL)
    adata = rename_var_to_gene_symbol(adata, ENSEMBL_RELEASE)
    adata = preprocess_data(adata)

    adata = label_cell_types(adata, LOW_RESOLUTION_MODEL, "celltype_low_resolution")
    plot_umap(adata, LOW_RES_PLOT_FILENAME, "celltype_low_resolution")

    adata = label_cell_types(adata, HIGH_RESOLUTION_MODEL, "celltype")
    plot_umap(adata, HIGH_RES_PLOT_FILENAME, "celltype")

    output_file_path = DATA_FILE_PATH.replace(".h5ad", "_labeled.h5ad")
    adata.write(output_file_path)


if __name__ == "__main__":
    main()
