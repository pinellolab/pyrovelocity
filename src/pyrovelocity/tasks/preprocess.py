import os
from pathlib import Path
from typing import List, Optional, Tuple

import anndata
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import seaborn as sns
from anndata._core.anndata import AnnData
from beartype import beartype
from scipy.sparse import issparse

from pyrovelocity.analysis.cytotrace import cytotrace_sparse
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._count_histograms import (
    plot_spliced_unspliced_histogram,
)
from pyrovelocity.tasks.data import load_anndata_from_path
from pyrovelocity.utils import ensure_numpy_array, print_anndata

from scipy.sparse import csr_matrix
from scipy.sparse import issparse
import warnings

__all__ = [
    "assign_colors",
    "compute_and_plot_qc",
    "copy_raw_counts",
    "get_high_us_genes",
    "get_thresh_histogram_title_from_path",
    "plot_high_us_genes",
    "preprocess_dataset"
]

logger = configure_logging(__name__)


@beartype
def preprocess_dataset(
    data_set_name: str,
    adata: str | Path | AnnData,
    data_processed_path: str | Path = "data/processed",
    reports_processed_path: str | Path = "reports/processed",
    overwrite: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 30,
    process_cytotrace: bool = False,
    use_obs_subset: bool = False,
    n_obs_subset: int = 300,
    use_vars_subset: bool = False,
    n_vars_subset: int = 200,
    count_threshold: int = 0,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    default_velocity_mode: str = "dynamical",
    vector_field_basis: str = "umap",
    cell_state: str = "clusters",
) -> Tuple[AnnData, Path, Path]:
    """
    Preprocess data.

    Args:
        data_set_name (str): name of data set.
        adata (AnnData, optional): AnnData object containing the data set to preprocess.
        data_processed_path (str | Path, optional): path to processed data. Defaults to "data/processed".
        overwrite (bool, optional): overwrite existing processed data. Defaults to False.
        n_top_genes (int, optional): number of genes to retain. Defaults to 2000.
        min_shared_counts (int, optional): minimum shared counts. Defaults to 30.
        process_cytotrace (bool, optional): process data with cytotrace. Defaults to False.
        use_obs_subset (bool, optional): use observation subset. Defaults to False.
        n_obs_subset (int, optional): number of observations to subset. Defaults to 300.
        use_vars_subset (bool, optional): use variable subset. Defaults to False.
        n_vars_subset (int, optional): number of variables to subset. Defaults to 200.
        count_threshold (int, optional): count threshold. Defaults to 0.
        n_pcs (int, optional): number of principal components. Defaults to 30.
        n_neighbors (int, optional): number of neighbors. Defaults to 30.
        default_velocity_mode (str, optional): default velocity mode. Defaults to "dynamical".
        vector_field_basis (str, optional): vector field basis. Defaults to "umap".
        cell_state (str, optional): Name of the cell state/cluster variable. Defaults to "clusters".

    Returns:
        AnnData: processed AnnData object

    Examples:
        >>> # xdoctest: +SKIP
        >>> from pyrovelocity.data import download_dataset
        >>> tmp = getfixture('tmp_path')
        >>> simulated_dataset_path = download_dataset(
        ...   'simulated',
        ...   str(tmp) + '/data/external',
        ...   'simulate',
        ...   n_obs=100,
        ...   n_vars=300,
        ... )
        >>> preprocess_dataset(
        ...     data_set_name="simulated",
        ...     adata=simulated_dataset_path,
        ... )
    """
    if isinstance(adata, str | Path):
        data_path = adata
        adata = load_anndata_from_path(data_path)
    else:
        data_path = "AnnData object"

    if use_obs_subset:
        if n_obs_subset > adata.n_obs:
            logger.warning(
                f"n_obs_subset: {n_obs_subset} > adata.n_obs: {adata.n_obs}\n"
                f"setting n_obs_subset to adata.n_obs: {adata.n_obs}"
            )
            n_obs_subset = adata.n_obs
        else:
            if n_obs_subset < 10:
                logger.warning(
                    f"n_obs_subset: {n_obs_subset} < 10\n"
                    f"setting n_obs_subset to 10"
                )
                n_obs_subset = 10
            logger.info(
                f"extracting {n_obs_subset} observation subset from {adata.n_obs} observations\n"
            )
            adata = adata[
                np.random.choice(
                    adata.obs.index,
                    size=n_obs_subset,
                    replace=False,
                ),
                :,
            ].copy()

    reports_processed_path = Path(reports_processed_path) / data_set_name
    logger.info(
        f"\n\nVerifying existence of path for:\n\n"
        f"  preprocessing reports: {reports_processed_path}\n"
    )
    reports_processed_path.mkdir(parents=True, exist_ok=True)
    processed_path = os.path.join(
        data_processed_path, f"{data_set_name}_processed.h5ad"
    )
    count_threshold_histogram_path = os.path.join(
        reports_processed_path, f"count_thresholded_histogram.pdf"
    )
    logger.info(
        f"\n\nVerifying existence of path for:\n\n"
        f"  processed data: {data_processed_path}\n"
    )
    Path(data_processed_path).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"\n\nPreprocessing {data_set_name} data :\n\n"
        f"  from: {data_path}\n"
        f"  to processed: {processed_path}\n"
    )

    print_anndata(adata)
    compute_and_plot_qc(
        adata=adata,
        qc_plots_path=reports_processed_path,
    )
    print_anndata(adata)

    if (
        os.path.isfile(processed_path)
        and os.access(processed_path, os.R_OK)
        and not overwrite
    ):
        logger.info(f"{processed_path} exists")
        return adata, Path(processed_path), reports_processed_path
    else:
        logger.info(f"generating {processed_path} ...")

        if "raw_unspliced" not in adata.layers:
            copy_raw_counts(adata)
            print_anndata(adata)

        splice_state_histogram = plot_spliced_unspliced_histogram(
            adata=adata,
            spliced_layer="raw_spliced",
            unspliced_layer="raw_unspliced",
            min_count=3,
            max_count=200,
        )
        splice_state_histogram.save(
            reports_processed_path / "splice_state_histogram.pdf"
        )
        splice_state_histogram.save(
            reports_processed_path / "splice_state_histogram.png"
        )
        splice_state_histogram.save(
            reports_processed_path / "splice_state_histogram.html"
        )

        if process_cytotrace:
            logger.info("processing data with cytotrace ...")
            cytotrace_sparse(adata, layer="spliced")

        adata_tmp = scv.pp.filter_and_normalize(
            adata,
            min_shared_counts=min_shared_counts,
            n_top_genes=n_top_genes,
            log=False,
            copy=True,
        )
        sc.pp.log1p(adata_tmp)
        if adata_tmp.n_vars < n_obs_subset:
            logger.warning(
                f"adata.n_vars: {adata_tmp.n_vars} < n_obs_subset: {n_obs_subset}\n"
                f"for data_set_name: {data_set_name} and min_shared_counts: {min_shared_counts}\n"
                f"setting min_shared_counts to min of 5 and {min_shared_counts}\n"
            )
            min_shared_counts = min(5, min_shared_counts)
            adata = scv.pp.filter_and_normalize(
                adata,
                min_shared_counts=min_shared_counts,
                n_top_genes=n_top_genes,
                log=False,
                copy=True,
            )
            sc.pp.log1p(adata)
            logger.warning(
                f"after updating min_shared_counts: {min_shared_counts},\n"
                f"adata.n_vars: {adata.n_vars}\n"
            )
        else:
            adata = adata_tmp.copy()
        del adata_tmp

        if adata.n_vars < n_top_genes:
            logger.warning(
                f"adata.n_vars: {adata.n_vars} < n_top_genes: {n_top_genes}\n"
                f"for data_set_name: {data_set_name} and min_shared_counts: {min_shared_counts}\n"
            )

        plot_high_us_genes(
            adata=adata,
            count_threshold_histogram_path=count_threshold_histogram_path,
            minlim_u=count_threshold,
            minlim_s=count_threshold,
            unspliced_layer="raw_unspliced",
            spliced_layer="raw_spliced",
        )
        adata = get_high_us_genes(
            adata,
            minlim_u=count_threshold,
            minlim_s=count_threshold,
            unspliced_layer="raw_unspliced",
            spliced_layer="raw_spliced",
        )

        if adata.n_vars <= n_pcs:
            logger.warning(
                f"adata.n_vars: {adata.n_vars} <= n_pcs: {n_pcs}\n"
                f"setting n_pcs to adata.n_vars - 1: {adata.n_vars -1}"
            )
            n_pcs = adata.n_vars - 1
        if "X_pca" not in adata.obsm.keys():
            sc.pp.pca(
                adata,
                svd_solver="arpack",
            )

        sc.pp.neighbors(
            adata=adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
        )
        scv.pp.moments(
            data=adata,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
        )
        scv.tl.recover_dynamics(
            data=adata,
            n_jobs=-1,
            use_raw=False,
            # show_progress_bar=False,
        )

        scv.tl.velocity(
            data=adata,
            mode=default_velocity_mode,
            use_raw=False,
        )

        # TODO: recompute umap for "larry_tips"
        # TODO: export QC plots, which will require use of the cell_state variable
        logger.info(f"cell state variable: {cell_state}")
        if "X_umap" not in adata.obsm.keys():
            scv.tl.umap(adata)
        if "leiden" not in adata.obs.keys():
            sc.tl.leiden(adata)
        if use_vars_subset:
            likelihood_sorted_genes = (
                adata.var["fit_likelihood"].sort_values(ascending=False).index
            )
            top_30_genes_list = likelihood_sorted_genes[:30].tolist()
            logger.info(f"\nTop 30 genes:\n{top_30_genes_list}\n\n")
            if n_vars_subset > adata.n_vars:
                logger.warning(
                    f"n_vars_subset: {n_vars_subset} > adata.n_vars: {adata.n_vars}\n"
                    f"setting n_vars_subset to adata.n_vars: {adata.n_vars}"
                )
                n_vars_subset = adata.n_vars
            if n_vars_subset > len(likelihood_sorted_genes):
                logger.warning(
                    f"n_vars_subset: {n_vars_subset} > len(likelihood_sorted_genes): {len(likelihood_sorted_genes)}\n"
                    f"setting n_vars_subset to len(likelihood_sorted_genes): {len(likelihood_sorted_genes)}"
                )
                n_vars_subset = len(likelihood_sorted_genes)
            adata = adata[:, likelihood_sorted_genes[:n_vars_subset]].copy()
        scv.tl.velocity_graph(
            data=adata,
            n_jobs=-1,
            # show_progress_bar=False,
        )

        scv.tl.velocity_embedding(adata, basis=vector_field_basis)

        scv.tl.latent_time(adata)

        print_anndata(adata)
        adata.write(processed_path)

        if os.path.isfile(processed_path) and os.access(
            processed_path, os.R_OK
        ):
            logger.info(f"successfully generated {processed_path}")
            return adata, Path(processed_path), reports_processed_path
        else:
            logger.error(f"cannot find and read {processed_path}")


@beartype
def compute_and_plot_qc(
    adata: AnnData,
    qc_plots_path: str | Path,
) -> Tuple[str, str, str]:
    """
    Compute and plot quality control metrics.

    Args:
        adata (str | Path | AnnData): AnnData object
        qc_plots_path (str | Path): path to save quality control plots

    Returns:
        Tuple[str, str, str]:
            paths to mitochondrial counts,
            ribosomal counts, and counts in observations plots
    """
    mitochondrial_counts_plot_path = os.path.join(
        qc_plots_path,
        "mitochondrial_counts_histogram.pdf",
    )
    ribosomal_counts_plot_path = os.path.join(
        qc_plots_path,
        "ribosomal_counts_histogram.pdf",
    )
    counts_in_obs_plot_path = os.path.join(
        qc_plots_path,
        "counts_in_obs.pdf",
    )

    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "Mt-", "mt-"))
    adata.var["ribo"] = adata.var_names.str.startswith(
        ("RPS", "Rps", "rps", "RPL", "Rpl", "rpl")
    )

    sc.pp.calculate_qc_metrics(
        adata=adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    ax = sns.histplot(
        adata.obs,
        x="pct_counts_mt",
        color="#ff6a14",
    )
    ax.set_xlabel("mitochondrial counts (%)")
    fig = ax.get_figure()
    for ext in ["", ".png"]:
        fig.savefig(
            f"{mitochondrial_counts_plot_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    ax = sns.histplot(
        adata.obs,
        x="pct_counts_ribo",
        color="#ff6a14",
    )
    ax.set_xlabel("ribosomal counts (%)")
    fig = ax.get_figure()
    for ext in ["", ".png"]:
        fig.savefig(
            f"{ribosomal_counts_plot_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    numeric_obs = adata.obs.copy()
    numeric_obs["n_genes_by_counts"] = pd.to_numeric(
        numeric_obs["n_genes_by_counts"],
        errors="coerce",
    )
    fig = sns.jointplot(
        data=numeric_obs,
        x="total_counts",
        y="n_genes_by_counts",
        color="#ff6a14",
        marginal_ticks=True,
        kind="scatter",
        alpha=0.4,
    )
    fig.plot_joint(
        sns.kdeplot,
        color="gray",
        alpha=0.6,
    )
    fig.set_axis_labels(
        xlabel="Total counts in cell",
        ylabel="Number of genes >=1 count in cell",
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{counts_in_obs_plot_path}{ext}",
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )
    return (
        mitochondrial_counts_plot_path,
        ribosomal_counts_plot_path,
        counts_in_obs_plot_path,
    )


def copy_raw_counts(
    adata: AnnData,
) -> AnnData:
    """
    Copy unspliced and spliced raw counts to adata.layers and adata.obs.

    Args:
        adata (AnnData): AnnData object

    Returns:
        AnnData: AnnData object with raw counts.

    Examples:
        >>> from pyrovelocity.utils import generate_sample_data
        >>> adata = generate_sample_data()
        >>> copy_raw_counts(adata)
    """
    adata.layers["raw_unspliced"] = adata.layers["unspliced"]
    logger.info(
        "'raw_unspliced' key added raw unspliced counts to adata.layers"
    )
    adata.layers["raw_spliced"] = adata.layers["spliced"]
    logger.info(
        "'raw_spliced' key added raw spliced counts added to adata.layers"
    )
    u_lib_size_raw = (
        adata.layers["raw_unspliced"].toarray().sum(-1)
        if issparse(adata.layers["raw_unspliced"])
        else adata.layers["raw_unspliced"].sum(-1)
    )
    adata.obs["u_lib_size_raw"] = u_lib_size_raw
    logger.info(
        f"'u_lib_size_raw' key added unspliced library size to adata.obs, total: {u_lib_size_raw.sum()}"
    )

    s_lib_size_raw = (
        adata.layers["raw_spliced"].toarray().sum(-1)
        if issparse(adata.layers["raw_spliced"])
        else adata.layers["raw_spliced"].sum(-1)
    )
    adata.obs["s_lib_size_raw"] = s_lib_size_raw
    logger.info(
        f"'s_lib_size_raw' key added spliced library size to adata.obs, total: {s_lib_size_raw.sum()}"
    )
    return adata


def assign_colors(
    max_spliced: int, max_unspliced: int, minlim_s: int, minlim_u: int
) -> List[str]:
    return [
        "black"
        if (spliced >= minlim_s) & (unspliced >= minlim_u)
        else "lightgrey"
        for spliced, unspliced in zip(max_spliced, max_unspliced)
    ]


def get_thresh_histogram_title_from_path(path):
    title = os.path.basename(path)
    title = os.path.splitext(title)[0]
    title = title.replace("_thresh_histogram", "")
    return title.replace("_", " ")


def plot_high_us_genes(
    adata: anndata.AnnData,
    count_threshold_histogram_path: str,
    minlim_u: int = 3,
    minlim_s: int = 3,
    unspliced_layer: str = "unspliced",
    spliced_layer: str = "spliced",
) -> Optional[matplotlib.figure.Figure]:
    if (
        adata is None
        or unspliced_layer not in adata.layers
        or spliced_layer not in adata.layers
    ):
        raise ValueError(
            "Invalid data set. Please ensure that adata is an AnnData object"
            "and that the layers 'unspliced' and 'spliced' are present."
        )

    max_unspliced = np.array(
        np.max(ensure_numpy_array(adata.layers[unspliced_layer]), axis=0)
    ).flatten()
    max_spliced = np.array(
        np.max(ensure_numpy_array(adata.layers[spliced_layer]), axis=0)
    ).flatten()

    # create figure
    x = max_spliced
    y = max_unspliced

    title = get_thresh_histogram_title_from_path(count_threshold_histogram_path)

    colors = assign_colors(max_spliced, max_unspliced, minlim_s, minlim_u)

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    fig = plt.figure(figsize=(4, 4))
    fig.suptitle(title, y=1.00, fontsize=12)

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # scatter plot
    ax.scatter(max_spliced, max_unspliced, s=1, c=colors)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("max. spliced counts")
    ax.set_ylabel("max. unspliced counts")
    if minlim_s > 0:
        ax.axhline(y=minlim_s - 1, color="r", linestyle="--")
    if minlim_u > 0:
        ax.axvline(x=minlim_u - 1, color="r", linestyle="--")

    # histograms
    bins = 50
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(0, np.log10(bins[-1]), len(bins))
    ax_histx.hist(x, bins=logbins)
    ax_histy.hist(y, bins=logbins, orientation="horizontal")

    for ext in ["", ".png"]:
        fig.savefig(
            f"{count_threshold_histogram_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )
    plt.close(fig)
    return fig


def get_high_us_genes(
    adata: AnnData,
    minlim_u: int = 0,
    minlim_s: int = 0,
    unspliced_layer: str = "unspliced",
    spliced_layer: str = "spliced",
) -> AnnData:
    """
    Function to select genes that have spliced and unspliced counts above a
    certain threshold. Genes of which the maximum u and s count is above a set
    threshold are selected. Threshold varies per dataset and influences the
    numbers of genes that are selected.

    Parameters
    ----------
    adata
        Annotated data matrix
    minlim_u: `int` (default: 3)
        Threshold above which the maximum unspliced counts of a gene should fall to be included in the
        list of high US genes.
    minlim_s: `int` (default: 3)
        Threshold above which the maximum spliced counts of a gene should fall to be included in the
        list of high US genes.
    unspliced_layer: `str` (default: 'unspliced')
        Name of layer that contains the unspliced counts.
    spliced_layer: `str` (default: 'spliced')
        Name of layer that contains the spliced counts.
    """
    logger.info(f"adata.shape before filtering: {adata.shape}")
    from scipy import sparse

    # test if layers are not sparse but dense
    for layer in [unspliced_layer, spliced_layer]:
        if sparse.issparse(adata.layers[layer]):
            adata.layers[layer] = adata.layers[layer].todense()

    # get high US genes
    u_genes = np.max(adata.layers[unspliced_layer], axis=0) >= minlim_u
    s_genes = np.max(adata.layers[spliced_layer], axis=0) >= minlim_s
    us_genes = adata.var_names[np.array(u_genes & s_genes).flatten()].values
    adata = adata[:, us_genes]
    for layer in [unspliced_layer, spliced_layer]:
        adata.layers[layer] = sparse.csr_matrix(adata.layers[layer])
    logger.info(f"adata.shape after filtering: {adata.shape}")
    return adata

@beartype
def compute_metacells(
    adata_rna: AnnData,
    adata_atac: AnnData,
    latent_key: str,
    celltype_key: Optional[str] = None,
    n_neighbors: int = 10,
    n_neighbors_metacell: int = 5,
    resolution: int = 50,   
    verbose: bool = True,
    merge_knn_graph: bool = True,
    merge_umap: bool = True,
    umap_key: Optional[str] = None
) -> Tuple[AnnData, AnnData]:
    """
    Computes metacells, using low-level clustering in a given a latent-space. By default, includes
    summing up RNA counts, ATAC counts and optionally includes averaging UMAP coordinates and computing
    a new knn-graph for meta-cells. If a celltype key is provided, metacells are named by their most 
    frequent celltype label.

    Args:
        adata_rna (AnnData): AnnData object with RNA counts.
        adata_atac (AnnData): AnnData object with ATAC counts.
        latent_key (str): Name of latent space key in .obsm slot in adata_rna, e.g. X_pca.
        celltype_key (Optional[str] optional): Name of cell type key in .obs column in adata_rna.
        n_neighbors (int, optional): Number of nearest neighbors to use in initial knn-graph used 
        for metacell construction. Defaults to 10.
        n_neighbors_metacell (int, optional): Number of nearest neighbors to use in new knn-graph
        for metacells. Defaults to 5.
        resolution (int, optional): Resolution at which to do leiden clustering for metacells.
        Defaults to 50. 
        verbose (bool, optional): Whether to print out progress and make diagnostic plots 
        of metacell computations. Defaults to True.
        merge_knn_graph (bool, optional): Whether to produce a new knn graph for metacells.
        merge_umap (bool, optional): Whether to produce a umap embedding for metacells from the previous embedding of cells.
        If no previous embedding is present it will be recomputed for the original anndata object.
        umap_key (str, optional): Key of UMAP embedding in adata_rna.obsm
    Returns:
        Tuple[AnnData, AnnData]: Tuple containing two anndata objects containing RNA and ATAC counts for metacells.

    Examples:
        >>> from pyrovelocity.tests.synthetic_AnnData import synthetic_AnnData
        >>> adata_rna = synthetic_AnnData(seed = 1)
        >>> adata_atac = synthetic_AnnData(seed = 2)
        >>> sc.tl.pca(adata_rna, n_comps=3)
        >>> compute_metacells(adata_rna, adata_atac,
                  latent_key = 'X_pca',
                  resolution = 1,
                  celltype_key = 'cell_type')
    """
    
    # Check input makes sense:
    if len(adata_rna.obs_names) != len(adata_atac.obs_names):
        raise ValueError("RNA and ATAC data do not contain the same cells number of cells.")
    if (adata_rna.obs_names != adata_atac.obs_names).all():
        raise ValueError("RNA and ATAC data do not contain the same cells in obs_names.")

    if not isinstance(adata_rna.X, csr_matrix):
        adata_rna.X = csr_matrix(adata_rna.X)
    if not isinstance(adata_atac.X, csr_matrix):
        adata_atac.X = csr_matrix(adata_atac.X)
    
    # Define functions for all processing steps:
    
    @beartype
    def merge_RNA(
    adata_rna: AnnData,
    cluster_key: str,
    celltype_key: Optional[str] = None,
    verbose: bool = True,
    )-> AnnData:

        if verbose:
            print('merging RNA counts')
        
        X = np.concatenate([np.sum(adata_rna.X[adata_rna.obs[cluster_key] == c,:], axis = 0) for c in np.unique(adata_rna.obs[cluster_key])], axis = 0)
        print(X.shape)
        adata_meta = sc.AnnData(X  = np.array(X))
        adata_meta.var = adata_rna.var
        adata_meta.obs['n_cells'] = [np.sum(adata_rna.obs[cluster_key] == c) for c in np.unique(adata_rna.obs[cluster_key])]
        if celltype_key:
            adata_meta.obs[celltype_key] = [adata_rna[adata_rna.obs[cluster_key] == c,:].obs[celltype_key].mode()[0] for c in np.unique(adata_rna.obs[cluster_key])]
        adata_meta.obs['RNA counts'] = np.sum(adata_meta.X, axis = 1)
        
        if 'unspliced' in adata_rna.layers:
            adata_meta.layers['unspliced'] = np.concatenate([np.sum(adata_rna.layers['unspliced'][adata_rna.obs[cluster_key] == c,:], axis = 0) for c in np.unique(adata_rna.obs[cluster_key])], axis = 0)
            adata_meta.layers['unspliced'] = csr_matrix(adata_meta.layers['unspliced'], dtype=np.uint16)
        if 'spliced' in adata_rna.layers:
            adata_meta.layers['spliced'] = np.concatenate([np.sum(adata_rna.layers['spliced'][adata_rna.obs[cluster_key] == c,:], axis = 0) for c in np.unique(adata_rna.obs[cluster_key])], axis = 0)                
            adata_meta.layers['spliced'] = csr_matrix(adata_meta.layers['spliced'], dtype=np.uint16)
        
        if verbose:
            print('Mean RNA counts per cell before: ', np.mean(adata_rna.obs['RNA counts']))
            print('Mean RNA counts per cell after: ', np.mean(adata_meta.obs['RNA counts']))
            plt.hist(adata_rna.obs['RNA counts'], bins = 10, label = 'single cells', alpha = 0.5)
            plt.hist(adata_meta.obs['RNA counts'],bins = 20, label = 'meta cells', alpha = 0.5)
            plt.xlabel('Total Counts')
            plt.ylabel('Occurences')
            plt.legend()
            plt.show()
            
        return adata_meta

    @beartype
    def merge_UMAP(
    adata_rna: AnnData,
    adata_meta: AnnData,
    cluster_key: str,
    umap_key: str = 'X_umap',
    verbose: bool = True,
    )-> AnnData:

        if verbose:
            print('merging UMAP')
        
        adata_meta.obsm[umap_key] = np.concatenate([np.expand_dims(np.mean(adata_rna.obsm[umap_key][adata_rna.obs[cluster_key] == c,:], axis = 0), axis = -1) for c in np.unique(adata_rna.obs[cluster_key])], axis = 1).T
            
        return adata_meta

    @beartype
    def merge_ATAC(
    adata_atac: AnnData,
    cluster_key: str,
    celltype_key: Optional[str] = None,
    verbose: bool = True
    )-> AnnData:

        if verbose:
                print('merging ATAC counts')
        
        X = np.concatenate([np.sum(adata_atac.X[adata_rna.obs[cluster_key] == c,:], axis = 0) for c in np.unique(adata_atac.obs[cluster_key])], axis = 0)
        adata_atac_meta = sc.AnnData(X  = np.array(X))
        adata_atac_meta.var = adata_atac.var
        adata_atac_meta.obs['n_cells'] = [np.sum(adata_atac.obs[cluster_key] == c) for c in np.unique(adata_atac.obs[cluster_key])]
        if celltype_key:
            adata_atac_meta.obs[celltype_key] = [adata_atac[adata_atac.obs[cluster_key] == c,:].obs[celltype_key].mode()[0] for c in np.unique(adata_atac.obs[cluster_key])]
        adata_atac_meta.obs['ATAC counts'] = np.sum(adata_atac_meta.X, axis = 1)
        
        if verbose:
            print('Mean ATAC counts per cell before: ', np.mean(adata_atac.obs['ATAC counts']))
            print('Mean ATAC counts per cell after: ', np.mean(adata_atac_meta.obs['ATAC counts']))
            plt.hist(adata_atac.obs['ATAC counts'], bins = 10, label = 'single cells', alpha = 0.5)
            plt.hist(adata_atac_meta.obs['ATAC counts'],bins = 20, label = 'meta cells', alpha = 0.5)
            plt.xlabel('Total Counts')
            plt.ylabel('Occurences')
            plt.legend()
            plt.show()
            
        return adata_atac_meta

    @beartype
    def merge_knn(
    adata_rna: AnnData,
    adata_meta: AnnData,
    cluster_key: str,
    n_neighbors: int = 6,
    verbose: bool = True
    ) -> AnnData:
            
        if verbose:
            print('merging knn graph')
            
        distance_matrix = (adata_rna.obsp['distances'].toarray() != 0)*1
        clusters = adata_rna.obs[cluster_key]
        for c in np.unique(clusters):
            subset = np.array(clusters == c)
            distance_matrix = np.concatenate([distance_matrix[~subset,:], np.expand_dims(np.sum(distance_matrix[subset,:], axis = 0), axis = 0)], axis = 0)
            distance_matrix = np.concatenate([distance_matrix[:,~subset], np.expand_dims(np.sum(distance_matrix[:,subset], axis = 1), axis = 1)], axis = 1)
            clusters = np.concatenate([clusters[~subset], np.expand_dims(np.array((c)), axis = 0)])
        adata_meta.obsm['N_cn'] = np.stack([np.argsort(-1*distance_matrix[i,:])[:n_neighbors+1] for i in range(len(distance_matrix[:,0]))], axis = 0)

        return adata_meta
    
    # Run through the metacell construction:
    adata_atac = adata_atac[adata_rna.obs_names,:]
    if celltype_key:
        adata_atac.obs[celltype_key] = adata_rna.obs[celltype_key]
    adata_rna.obs['RNA counts'] = np.sum(adata_rna.X, axis = 1)
    adata_atac.obs['ATAC counts'] = np.sum(adata_atac.X, axis = 1)
    
    if verbose:
        print('low resolution clustering cells')
    sc.pp.neighbors(adata_rna, use_rep=latent_key, n_neighbors=n_neighbors)
    cluster_key = "leiden"   
    sc.tl.leiden(adata_rna, key_added=cluster_key, resolution=resolution)
    adata_atac.obs[cluster_key] = adata_rna.obs[cluster_key]
    
    if verbose:
        print('total number of cells', len(adata_rna.obs_names))
        print('total number of meta-cells', len(np.unique(adata_rna.obs[cluster_key])))
        print('minimum cells/meta-cell', np.min(adata_rna.obs[cluster_key].value_counts()))
        print('average cells/meta-cell', np.round(np.mean(adata_rna.obs[cluster_key].value_counts()),1))
        print('maximum cells/meta_cell', np.max(adata_rna.obs[cluster_key].value_counts()))
        plt.hist(adata_rna.obs[cluster_key].value_counts(), bins = 10)
        plt.xlabel('cells per meta-cell')
        plt.ylabel('number of meta-cells')
    
    adata_meta = merge_RNA(adata_rna, verbose = verbose,
              cluster_key = cluster_key, celltype_key = celltype_key)    
    
    if merge_umap:
        if not umap_key:
            sc.tl.umap(adata_rna)
            umap_key = 'X_umap'
        elif umap_key not in adata_rna.obsm:
            warnings.warn("Umap_key not found in AnnData. Computing with sc.tl.umap()...", category=UserWarning)
            sc.tl.umap(adata_rna)
            umap_key = 'X_umap'
        adata_meta = merge_UMAP(adata_rna, adata_meta, umap_key = umap_key,verbose = verbose, cluster_key = cluster_key)
    
    adata_atac_meta = merge_ATAC(adata_atac, verbose = verbose,
              cluster_key = cluster_key, celltype_key = celltype_key)
    
    adata_meta.obs['ATAC counts'] = adata_atac_meta.obs['ATAC counts']
    
    if merge_knn_graph:
        
        adata_meta = merge_knn(
        adata_rna,
        adata_meta,
        cluster_key = cluster_key,
        n_neighbors = n_neighbors_metacell,
        verbose = True)
        
    if verbose:
        print('Done.')
    
    return adata_meta, adata_atac_meta

# ------------------------------------------------------------------------------


# TODO: remove deprecated function
# def load_pbmc68k(
#     data: str = "pbmc68k",
#     count_threshold: int = 0,
#     count_threshold_histogram_path: str = None,
# ) -> AnnData:
#     if data == "pbmc68k":
#         adata = pyrovelocity.datasets.pbmc68k()
#     elif os.path.isfile(data) and os.access(data, os.R_OK):
#         adata = sc.read(data)

#     print_anndata(adata)
#     if "raw_unspliced" not in adata.layers:
#         copy_raw_counts(adata)
#         print_anndata(adata)
#     # Integrated into pyrovelocity.datasets.pbmc68k()
#     # print("Removing duplicate cells and tSNE x-parity in pbmc68k data...")
#     # scv.pp.remove_duplicate_cells(adata)
#     # adata.obsm["X_tsne"][:, 0] *= -1
#     scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
#     plot_high_us_genes(
#         adata,
#         count_threshold_histogram_path,
#         minlim_u=count_threshold,
#         minlim_s=count_threshold,
#         unspliced_layer="raw_unspliced",
#         spliced_layer="raw_spliced",
#     )
#     adata = get_high_us_genes(
#         adata,
#         minlim_u=count_threshold,
#         minlim_s=count_threshold,
#         unspliced_layer="raw_unspliced",
#         spliced_layer="raw_spliced",
#     )
#     scv.pp.moments(adata)
#     scv.tl.velocity(adata, mode="stochastic")
#     scv.tl.recover_dynamics(adata, n_jobs=-1)

#     return adata


# TODO: remove deprecated function
# def load_data(
#     data: str = "pancreas",
#     processed_path: str = None,
#     count_threshold: int = 0,
#     count_threshold_histogram_path: str = None,
# ) -> AnnData:
#     """
#     Preprocess data from scvelo.

#     Args:
#         data (str, optional): data set name. Defaults to scvelo's "pancreas" data set.
#         top_n (int, optional): number of genes to retain. Defaults to 2000.
#         min_shared_counts (int, optional): minimum shared counts. Defaults to 30.
#         eps (float, optional): tolerance. Defaults to 1e-6.
#         force (bool, optional): force reprocessing. Defaults to False.
#         processed_path (str, optional): path to read/write processed AnnData. Defaults to None.

#     Returns:
#         AnnData: processed AnnData object
#     """
#     print("Dataset name:", data)
#     if data == "pancreas":
#         adata = pyrovelocity.datasets.pancreas()
#     elif data == "bonemarrow":
#         adata = pyrovelocity.datasets.bonemarrow()
#     elif data == "pbmc68k":
#         adata = pyrovelocity.datasets.pbmc68k()
#     elif data == "pons":
#         adata = pyrovelocity.datasets.pons()
#     elif data == "pbmc5k":
#         adata = pyrovelocity.datasets.pbmc5k()
#     elif data == "pbmc10k":
#         adata = pyrovelocity.datasets.pbmc10k()
#     elif data == "larry":
#         adata = pyrovelocity.datasets.larry()
#     elif data == "larry_mono":
#         adata = pyrovelocity.datasets.larry_mono()
#     elif data == "larry_neu":
#         adata = pyrovelocity.datasets.larry_neu()
#     elif data == "larry_multilineage":
#         adata = pyrovelocity.datasets.larry_multilineage()
#     elif data == "larry_tips":
#         adata = pyrovelocity.datasets.larry_tips()
#     elif data == "larry_cospar":
#         adata = pyrovelocity.datasets.larry_cospar()
#     elif data == "larry_cytotrace":
#         adata = pyrovelocity.datasets.larry_cytotrace()
#     elif data == "larry_dynamical":
#         adata = pyrovelocity.datasets.larry_dynamical()
#     print_anndata(adata)
#     return adata


# TODO: remove deprecated function
# @beartype
# def preprocess_data(
#     data_path: str | Path = "data/external/simulated.h5ad",
#     # process_args: dict = {"count_threshold": 0},
#     preprocess_data_args: PreprocessDataInterface = PreprocessDataInterface(),
#     data_processed_path: str | Path = "data/processed",
# ) -> Path:
#     """
#     Preprocess dataset.

#     Args:
#         data_path (str, optional): _description_. Defaults to "data/external/simulated.h5ad".
#         process_args (_type_, optional): _description_. Defaults to {"count_thres": 0}.
#         data_processed_path (str, optional): _description_. Defaults to "data/processed".

#     Returns:
#         Path: path to processed dataset.

#     Examples:
#         >>> from pyrovelocity.data import download_dataset, subset
#         >>> from pyrovelocity.preprocess import preprocess_dataset
#         >>> pancreas_path = download_dataset(
#         ...     data_set_name="pancreas",
#         ... ) # xdoctest: +SKIP
#         >>> preprocess_dataset(
#         ...     data_path=pancreas_path,
#         ... ) # xdoctest: +SKIP
#         >>> pbmc68k_path = download_dataset(
#         ...     data_set_name="pbmc68k",
#         ... ) # xdoctest: +SKIP
#         >>> _, subset_output_path = subset(
#         ...    file_path=pbmc68k_path,
#         ...    n_obs=1000,
#         ...    save_subset=True,
#         ... ) # xdoctest: +SKIP
#         >>> preprocessed_path = preprocess_dataset(
#         ...     data_path=subset_output_path
#         ... ) # xdoctest: +SKIP
#     """
#     data_set_name = Path(data_path).stem
#     processed_path = os.path.join(
#         data_processed_path, f"{data_set_name}_processed.h5ad"
#     )
#     count_threshold_histogram_path = os.path.join(
#         data_processed_path, f"{data_set_name}_thresh_histogram.pdf"
#     )
#     logger.info(
#         f"\n\nVerifying existence of path for:\n\n"
#         f"  processed data: {data_processed_path}\n"
#     )
#     Path(data_processed_path).mkdir(parents=True, exist_ok=True)

#     logger.info(
#         f"\n\nPreprocessing {data_set_name} data :\n\n"
#         f"  from external: {data_path}\n"
#         f"  to processed: {processed_path}\n"
#     )

#     if os.path.isfile(processed_path) and os.access(processed_path, os.R_OK):
#         logger.info(f"{processed_path} exists")
#         return Path(processed_path)
#     else:
#         logger.info(f"generating {processed_path} ...")
#         # adata, output_processed_path = preprocess_data(
#         #     data_set_name=data_set_name,
#         #     data_path=data_path,
#         #     processed_path=processed_path,
#         #     count_threshold_histogram_path=count_threshold_histogram_path,
#         #     **process_args,
#         # )
#         adata = preprocess_data(**asdict(preprocess_data_args))
#         adata.write(processed_path)
#         print_attributes(adata)

#         if os.path.isfile(processed_path) and os.access(
#             processed_path, os.R_OK
#         ):
#             logger.info(f"successfully generated {processed_path}")
#             return Path(processed_path)
#         else:
#             logger.error(f"cannot find and read {processed_path}")
#             logger.info(f"successfully generated {processed_path}")
#             return Path(processed_path)
#         else:
#             logger.error(f"cannot find and read {processed_path}")
