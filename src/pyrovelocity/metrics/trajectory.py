"""
Trajectory evaluation metrics for velocity models.

This module contains metrics for evaluating velocity model trajectories,
including directional correctness and coherence measures taken from:

> Qiao C, Huang Y. Representation learning of RNA velocity reveals robust cell
> transitions. Proc Natl Acad Sci U S A. 2021;118. doi:10.1073/pnas.2105859118.
"""

import numpy as np
from anndata import AnnData
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


@beartype
def keep_type(
    adata, nodes: np.ndarray, target: str, k_cluster: str
) -> np.ndarray:
    """
    Select cells of given target type.

    This implementation is included as a dependency of cross_boundary_correctness.
    It is based on the original implementation:
    https://github.com/qiaochen/VeloAE/blob/v0.2.0/veloproj/eval_util.py#L28-L41

    Args:
        adata: AnnData object
        nodes: Indexes for cells
        target: Cluster name
        k_cluster: Cluster key in adata.obs dataframe

    Returns:
        Selected cells matching the target cluster
    """
    return nodes[adata.obs[k_cluster][nodes].values == target]


@beartype
def cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    cluster_edges: List[Tuple[str, str]],
    k_velocity: str = "velocity",
    return_raw: bool = False,
    x_emb: str = "X_umap",
) -> Union[
    Dict[Tuple[str, str], List[float]],
    Tuple[Dict[Tuple[str, str], float], float],
]:
    """
    Cross-Boundary Direction Correctness Score.

    Calculates how well velocity vectors point toward neighboring cells in
    adjacent clusters, measuring the model's ability to predict correct
    developmental trajectories.

    Qiao C, Huang Y. Representation learning of RNA velocity reveals robust cell
    transitions. Proc Natl Acad Sci U S A. 2021;118. doi:10.1073/pnas.2105859118

    This implementation is based on the original implementation:
    https://github.com/qiaochen/VeloAE/blob/v0.2.0/veloproj/eval_util.py#L146-L200

    Args:
        adata: AnnData object
        k_cluster: Key to the cluster column in adata.obs DataFrame
        cluster_edges: Pairs of clusters with transition direction A->B
        k_velocity: Key to the velocity matrix in adata.obsm
        return_raw: Whether to return raw cell scores or aggregated scores
        x_emb: Key to embedding for visualization

    Returns:
        Raw cell scores by cluster edge or mean scores by cluster edge and overall mean
    """
    if "neighbors" not in adata.uns:
        raise ValueError("AnnData object must have neighbors computed")

    if "indices" not in adata.uns["neighbors"]:
        k = adata.uns["neighbors"]["params"]["n_neighbors"]
        connectivities = adata.obsp["connectivities"]

        if sparse.issparse(connectivities):
            connectivities_array = connectivities.toarray()
        else:
            connectivities_array = connectivities

        neighbor_indices = np.argsort(-connectivities_array, axis=1)[:, :k]
        adata.uns["neighbors"]["indices"] = neighbor_indices

    scores = {}
    all_scores = {}

    x_emb_data = adata.obsm[x_emb]

    if x_emb == "X_umap":
        v_emb = adata.obsm[f"{k_velocity}_umap"]
    else:
        v_emb = adata.obsm[
            [key for key in adata.obsm if key.startswith(k_velocity)][0]
        ]

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns["neighbors"]["indices"][sel]  # [n * 30]

        boundary_nodes = map(
            lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs
        )
        x_points = x_emb_data[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb_data[nodes] - x_pos
            dir_scores = cosine_similarity(
                position_dif, x_vel.reshape(1, -1)
            ).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score) if type_score else 0.0
        all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()]) if scores else 0.0


@beartype
def inner_cluster_coherence(
    adata, k_cluster: str, k_velocity: str, return_raw: bool = False
) -> Union[Dict[str, List[float]], Tuple[Dict[str, float], float]]:
    """
    In-cluster Coherence Score.

    Measures how aligned velocity vectors are within the same cluster,
    indicating consistency in predicted cellular trajectories.

    Qiao C, Huang Y. Representation learning of RNA velocity reveals robust cell
    transitions. Proc Natl Acad Sci U S A. 2021;118. doi:10.1073/pnas.2105859118

    This implementation is based on the original implementation:
    https://github.com/qiaochen/VeloAE/blob/v0.2.0/veloproj/eval_util.py#L203-L237

    Args:
        adata: AnnData object
        k_cluster: Key to the cluster column in adata.obs DataFrame
        k_velocity: Key to the velocity matrix in adata.layers
        return_raw: Whether to return raw scores or aggregated scores

    Returns:
        Raw scores by cluster or mean scores by cluster and overall mean
    """
    if "neighbors" not in adata.uns:
        raise ValueError("AnnData object must have neighbors computed")

    if "indices" not in adata.uns["neighbors"]:
        k = adata.uns["neighbors"]["params"]["n_neighbors"]
        connectivities = adata.obsp["connectivities"]

        if sparse.issparse(connectivities):
            connectivities_array = connectivities.toarray()
        else:
            connectivities_array = connectivities

        neighbor_indices = np.argsort(-connectivities_array, axis=1)[:, :k]
        adata.uns["neighbors"]["indices"] = neighbor_indices

    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns["neighbors"]["indices"][sel]
        same_cat_nodes = map(
            lambda nodes: keep_type(adata, nodes, cat, k_cluster), nbs
        )
        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]

        cat_score = [
            cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
            for ith, nodes in enumerate(same_cat_nodes)
            if len(nodes) > 0
        ]

        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score) if cat_score else 0.0

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()]) if scores else 0.0
