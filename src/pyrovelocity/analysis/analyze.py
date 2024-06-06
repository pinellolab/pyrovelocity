from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import sklearn
import umap
from anndata import AnnData
from astropy.stats import rayleightest
from beartype import beartype
from numpy import ndarray
from sklearn.pipeline import Pipeline

from pyrovelocity.analysis.cytotrace import compute_similarity2
from pyrovelocity.logging import configure_logging
from pyrovelocity.utils import ensure_numpy_array

__all__ = [
    "compute_mean_vector_field",
    "compute_volcano_data",
    "mae_per_gene",
    "vector_field_uncertainty",
]

logger = configure_logging(__name__)


def compute_mean_vector_field(
    posterior_samples,
    adata,
    basis="umap",
    n_jobs=1,
    spliced="spliced_pyro",
    raw=False,
):
    logger.info("Computing mean vector field")
    # scv.pp.neighbors(adata, use_rep="pca")
    if "X_pca" not in adata.obsm.keys():
        sc.pp.pca(
            data=adata,
            svd_solver="arpack",
        )
    sc.pp.neighbors(adata=adata, n_neighbors=30, use_rep="X_pca")

    adata.var["velocity_genes"] = True

    if spliced == "spliced_pyro":
        if raw:
            ut = posterior_samples["ut"]
            st = posterior_samples["st"]
            ut = ut / ut.sum(axis=-1, keepdims=True)
            st = st / st.sum(axis=-1, keepdims=True)
        else:
            ut = posterior_samples["ut"]
            st = posterior_samples["st"]
        adata.layers["spliced_pyro"] = st.mean(0).squeeze()
        # if ('u_scale' in posterior_samples) and ('s_scale' in posterior_samples)
        # TODO: two scale for Normal distribution
        if (
            "u_scale" in posterior_samples
        ):  # only one scale for Poisson distribution
            adata.layers["velocity_pyro"] = (
                ut * posterior_samples["beta"] / posterior_samples["u_scale"]
                - st * posterior_samples["gamma"]
            ).mean(0)
        else:
            if "beta_k" in posterior_samples:
                adata.layers["velocity_pyro"] = (
                    (
                        ut * posterior_samples["beta_k"]
                        - posterior_samples["st"] * posterior_samples["gamma_k"]
                    )
                    .mean(0)
                    .squeeze()
                )
            else:
                adata.layers["velocity_pyro"] = (
                    ut * posterior_samples["beta"]
                    - posterior_samples["st"] * posterior_samples["gamma"]
                ).mean(0)
        scv.tl.velocity_graph(
            adata, vkey="velocity_pyro", xkey="spliced_pyro", n_jobs=n_jobs
        )
    elif spliced in ["Ms"]:
        ut = adata.layers["Mu"]
        st = adata.layers["Ms"]
        if ("u_scale" in posterior_samples) and (
            "s_scale" in posterior_samples
        ):
            adata.layers["velocity_pyro"] = (
                ut
                * posterior_samples["beta"]
                / (posterior_samples["u_scale"] / posterior_samples["s_scale"])
                - st * posterior_samples["gamma"]
            ).mean(0)
        else:
            adata.layers["velocity_pyro"] = (
                ut * posterior_samples["beta"]
                - posterior_samples["st"] * posterior_samples["gamma"]
            ).mean(0)
        scv.tl.velocity_graph(
            adata, vkey="velocity_pyro", xkey="Ms", n_jobs=n_jobs
        )
    elif spliced in ["spliced"]:
        ut = adata.layers["unspliced"]
        st = adata.layers["spliced"]
        if ("u_scale" in posterior_samples) and (
            "s_scale" in posterior_samples
        ):
            adata.layers["velocity_pyro"] = (
                ut
                * posterior_samples["beta"]
                / (posterior_samples["u_scale"] / posterior_samples["s_scale"])
                - st * posterior_samples["gamma"]
            ).mean(0)
        else:
            adata.layers["velocity_pyro"] = (
                ut * posterior_samples["beta"]
                - posterior_samples["st"] * posterior_samples["gamma"]
            ).mean(0)
        scv.tl.velocity_graph(
            adata, vkey="velocity_pyro", xkey="spliced", n_jobs=n_jobs
        )

    scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis=basis)


@beartype
def compute_volcano_data(
    posterior_samples: List[Dict[str, ndarray]],
    adata: List[AnnData],
    time_correlation_with: str = "s",
    selected_genes: Optional[List[str]] = None,
    negative: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    assert isinstance(posterior_samples, (tuple, list))
    assert isinstance(adata, (tuple, list))
    assert "s" in posterior_samples[0]
    assert "alpha" in posterior_samples[0]

    maes_list = []
    cors = []
    genes = []
    labels = []
    switching = []
    for p, ad, label in zip(posterior_samples, adata, ["train", "valid"]):
        print(label)
        for sample in range(p["alpha"].shape[0]):
            maes_list.append(
                mae_per_gene(
                    p["s"][sample].squeeze(),
                    ensure_numpy_array(ad.layers["raw_spliced"]),
                )
            )
            df_genes_cors = compute_similarity2(
                p[time_correlation_with][sample].squeeze(),
                p["cell_time"][sample].squeeze().reshape(-1, 1),
            )
            cors.append(df_genes_cors[0])
            genes.append(ad.var_names.values)
            labels.append([f"Poisson_{label}"] * len(ad.var_names.values))

    volcano_data = pd.DataFrame(
        {
            "mean_mae": np.hstack(maes_list),
            "label": np.hstack(labels),
            "time_correlation": np.hstack(cors),
            "genes": np.hstack(genes),
        }
    )
    volcano_data = volcano_data.groupby("genes").mean(
        ["mean_mae", "time_correlation"]
    )

    volcano_data.loc[:, "mean_mae_rank"] = volcano_data.mean_mae.rank(
        ascending=False
    )
    volcano_data.loc[
        :, "time_correlation_rank"
    ] = volcano_data.time_correlation.apply(abs).rank(ascending=False)
    volcano_data.loc[:, "rank_product"] = (
        volcano_data.mean_mae_rank * volcano_data.time_correlation_rank
    )

    if selected_genes is None:
        genes = (
            volcano_data.sort_values("mean_mae", ascending=False)
            .head(300)
            .sort_values("time_correlation", ascending=negative)
            .head(4)
            .index.tolist()
        )
    else:
        genes = selected_genes
    volcano_data.loc[:, "selected genes"] = 0
    volcano_data.loc[genes, "selected genes"] = 1

    return volcano_data, genes


def vector_field_uncertainty(
    adata: AnnData,
    posterior_samples: dict[str, ndarray],
    basis: str = "tsne",
    n_jobs: int = 1,
    denoised: bool = False,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Run cosine similarity-based vector field across posterior samples"""

    logger.info("Estimating vector field uncertainty")
    # fig, ax = plt.subplots(10, 3)
    # fig.set_size_inches(16, 36)
    # ax = ax.flatten()
    v_map_all = []
    if ("u_scale" in posterior_samples) and (
        "s_scale" in posterior_samples
    ):  # Gaussian models
        scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
    elif ("u_scale" in posterior_samples) and not (
        "s_scale" in posterior_samples
    ):  # Poisson Model 2
        scale = posterior_samples["u_scale"]
    else:  # Poisson Model 1
        scale = 1

    if "beta_k" in posterior_samples:
        velocity_samples = (
            posterior_samples["ut"] * posterior_samples["beta_k"] / scale
            - posterior_samples["st"] * posterior_samples["gamma_k"]
        )
    else:
        velocity_samples = (
            posterior_samples["beta"] * posterior_samples["ut"] / scale
            - posterior_samples["gamma"] * posterior_samples["st"]
        )
    if denoised:
        projection = [
            (
                "PCA",
                sklearn.decomposition.PCA(random_state=99, n_components=50),
            ),
            ("UMAP", umap.UMAP(random_state=99, n_components=2)),
        ]
        pipelines = Pipeline(projection)
        expression = [posterior_samples["st"].mean(0)]
        pipelines.fit(expression[0])
        umap_orig = pipelines.transform(expression[0])
        adata.obsm["X_umap1"] = umap_orig
        joint_pcs = pipelines.steps[0][1].transform(expression[0])
        adata.obsm["X_pyropca"] = joint_pcs
        # scv.pp.neighbors(adata, use_rep="pyropca")
        sc.pp.neighbors(adata=adata, n_neighbors=30, use_rep="X_pyropca")
    else:
        # scv.pp.neighbors(adata, use_rep="pca")
        if "X_pca" not in adata.obsm.keys():
            sc.pp.pca(
                data=adata,
                svd_solver="arpack",
            )
        sc.pp.neighbors(adata=adata, n_neighbors=30, use_rep="X_pca")

    assert len(posterior_samples["st"].shape) == 3
    adata.var["velocity_genes"] = True
    for sample in range(posterior_samples["st"].shape[0]):
        adata.layers["spliced_pyro"] = posterior_samples["st"][sample]
        adata.layers["velocity_pyro"] = velocity_samples[sample]

        if basis == "pca":
            sc.pp.pca(
                data=adata,
                svd_solver="arpack",
            )
            scv.tl.velocity_embedding(
                adata,
                vkey="velocity_pyro",
                basis="pca",
                direct_pca_projection=True,
            )
        else:
            scv.tl.velocity_graph(
                adata, vkey="velocity_pyro", xkey="spliced_pyro", n_jobs=n_jobs
            )
            scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis=basis)
        v_map_all.append(adata.obsm[f"velocity_pyro_{basis}"])
    v_map_all = np.stack(v_map_all)
    embeds_radian = np.arctan2(v_map_all[:, :, 1], v_map_all[:, :, 0])
    from statsmodels.stats.multitest import multipletests

    rayleightest_pval = rayleightest(embeds_radian, axis=-2)
    _, fdri, _, _ = multipletests(rayleightest_pval, method="fdr_bh")
    return v_map_all, embeds_radian, fdri


def mae_per_gene(pred_counts: ndarray, true_counts: ndarray) -> ndarray:
    """Computes mean average error between counts and predicted probabilities."""
    error = np.abs(true_counts - pred_counts).sum(-2)
    total = np.clip(true_counts.sum(-2), 1, np.inf)
    return -np.array(error / total)


@beartype
def pareto_frontier_genes(
    volcano_data: pd.DataFrame,
    num_genes: int,
    max_iters: int = 2000,
) -> List[str]:
    """
    Identify genes on the Pareto frontier of a volcano plot with respect to MAE
    and time correlation.

    Args:
        volcano_data (pd.DataFrame): DataFrame containing MAE and time correlation
        num_genes (int): Number of genes to return.
        max_iters (int, optional): Maximum number of iterations. Defaults to 2000.

    Returns:
        List[str]: List of gene indices from volcano_data.
    """
    volcano_data = volcano_data.loc[
        ~volcano_data.index.str.contains(("^Rpl|^Rps"), case=False)
    ]
    pareto_frontier = pd.DataFrame()

    if len(volcano_data) < num_genes:
        logger.info(
            f"\nWarning: Not enough genes in the input data.\n"
            f"Only {len(volcano_data)} genes were found,\n"
            f"but {num_genes} were requested.\n"
            "Attempting to return as many as possible.\n"
        )

    pareto_frontier_rows = []

    counter = 0

    while (
        len(pareto_frontier_rows) < num_genes
        and len(volcano_data) > 0
        and counter < max_iters
    ):
        counter += 1
        sorted_data = volcano_data.sort_values(
            by=["mean_mae", "time_correlation"], ascending=[False, False]
        )
        pareto_frontier_current = [sorted_data.iloc[[0]]]

        for i in range(1, len(sorted_data)):
            if (
                sorted_data["time_correlation"].iloc[i]
                >= pareto_frontier_current[-1]["time_correlation"].iloc[0]
            ):
                pareto_frontier_current.append(sorted_data.iloc[[i]])
        pareto_frontier_indices = [
            pf.index[0] for pf in pareto_frontier_current
        ]
        volcano_data = volcano_data.drop(pareto_frontier_indices)
        pareto_frontier_rows.extend(pareto_frontier_current)

        logger.info(
            f"\nFound {len(pareto_frontier_current)} genes on the current Pareto frontier:\n\n"
            f"  {pareto_frontier_indices}\n\n"
            f"Genes identified thus far: {len(pareto_frontier_rows)}.\n"
            f"Number of iterations: {counter}.\n"
            f"Number of genes remaining: {len(volcano_data)}.\n\n"
        )
        if counter >= max_iters:
            logger.warning(
                f"\nWarning: Maximum number of iterations reached ({max_iters}).\n"
                f"Returning {len(pareto_frontier)} genes found so far.\n\n"
            )

    pareto_frontier = pd.concat(
        pareto_frontier_rows[:num_genes]
    ).drop_duplicates()
    pareto_frontier = pareto_frontier.sort_values(
        by="time_correlation", ascending=False
    )

    gene_indices = pareto_frontier.index.tolist()
    logger.info(
        f"\nFound {len(pareto_frontier)} genes on the Pareto frontier for {num_genes} requested:\n\n"
        f"  {gene_indices}\n\n"
    )
    return gene_indices
