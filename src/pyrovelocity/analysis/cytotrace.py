from time import time

import numpy as np
from anndata import AnnData
from beartype import beartype
from beartype.typing import Dict
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata

__all__ = [
    "compute_similarity2",
    "compute_similarity1",
    "compute_gcs",
    "threshold_and_normalize_similarity_matrix",
    "diffused",
    "cytotrace_sparse",
]


# TODO: refactor to subpackage exposing a minimal public interface
# TODO: migrate all justifiably eager imports to the tops of new modules
# TODO: enable runtime type-checking
# TODO: remove unused comments and print statements
# TODO: add unit tests


@beartype
def compute_similarity2(O: ndarray, P: ndarray) -> ndarray:
    """
    Compute pearson correlation between two matrices O and P.

    Args:
        O (ndarray): matrix of shape (n, t)
        P (ndarray): matrix of shape (n, m)

    Returns:
        ndarray: correlation matrix of shape (t, m)
    """
    # n traces of t samples
    (n, t) = O.shape

    # n predictions for each of m candidates
    (n_bis, m) = P.shape

    # compute O - mean(O)
    DO = O - (np.einsum("nt->t", O, optimize="optimal") / np.double(n))

    # compute P - mean(P)
    DP = P - (np.einsum("nm->m", P, optimize="optimal") / np.double(n))

    cov = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")

    varP = np.einsum("nm,nm->m", DP, DP, optimize="optimal")
    varO = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    tmp = np.einsum("m,t->mt", varP, varO, optimize="optimal")
    return cov / np.sqrt(tmp)


@beartype
def compute_similarity1(A: ndarray) -> ndarray:
    """
    Compute pairwise correlation of all columns in matrices A

    adapted from https://github.com/ikizhvatov/efficient-columnwise-correlation/blob/master/columnwise_corrcoef_perf.py

    Args:
        A (ndarray): matrix of shape (n, t)

    Returns:
        ndarray: correlation matrix of shape (t, t)
    """
    # genes x samples
    n, t = A.shape

    # compute O - mean(O)
    DO = A - (np.einsum("nt->t", A, optimize="optimal") / np.double(n))
    cov = np.einsum("nm,nt->mt", DO, DO, optimize="optimal")
    varP = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    return cov / np.sqrt(np.einsum("m,t->mt", varP, varP, optimize="optimal"))


@beartype
def compute_gcs(
    mat: ndarray,
    count: ndarray,
    top_n_genes: int = 200,
) -> ndarray:
    """
    Compute gene set enrichment scores by correlating gene count and gene expression

    Args:
        mat (ndarray): Matrix of shape (n_genes, n_cells)
        count (ndarray): Gene count
        top_n_genes (int, optional): Number of genes to select. Defaults to 200.

    Returns:
        ndarray:
    """
    corrs = compute_similarity2(mat.T, count.reshape(-1, 1))[0, :]
    corrs[np.isnan(corrs)] = -1
    gcs = mat[np.argsort(corrs)[::-1][:top_n_genes], :].mean(axis=0)
    return gcs


@beartype
def threshold_and_normalize_similarity_matrix(sim: ndarray) -> ndarray:
    """
    Transform a dense similarity matrix into a sparse, normalized version.

    Args:
        sim (ndarray): Similarity matrix

    Returns:
        ndarray: Thresholded and normalized similarity matrix
    """
    Ds = np.copy(sim)

    cutoff = np.mean(Ds)
    np.fill_diagonal(Ds, 0)

    Ds[Ds <= 0] = 0
    Ds[Ds <= cutoff] = 0

    zero_rows = Ds.sum(axis=1) == 0
    zero_cols = Ds.sum(axis=0) == 0

    Ds[~zero_rows, :] = (Ds[~zero_rows, :].T / Ds[~zero_rows, :].sum(axis=1)).T
    Ds[:, zero_cols] = 0
    return Ds


@beartype
def diffused(
    markov: ndarray,
    gcs: ndarray,
    ALPHA: float = 0.9,
) -> ndarray:
    """Compute diffused gene set enrichment scores

    Args:
        markov (ndarray): Markov state transition matrix
        gcs (ndarray): gene set enrichment scores
        ALPHA (float, optional): Defaults to 0.9.

    Returns:
        ndarray: _description_
    """
    v_prev = np.copy(gcs)
    v_curr = np.copy(gcs)

    for i in range(10000):
        v_prev = np.copy(v_curr)
        v_curr = ALPHA * (markov.dot(v_curr)) + (1 - ALPHA) * gcs

        diff = np.mean(np.abs(v_curr - v_prev))
        if diff <= 1e-6:
            break
    return v_curr


@beartype
def cytotrace_sparse(
    adata: AnnData,
    layer: str = "raw",
    cell_count: int = 20,
    top_n_features: int = 200,
    skip_regress: bool = False,
) -> Dict[str, ndarray]:
    "optimized version"
    proc_time = time()
    if not issparse(adata.layers[layer]):
        raise NotImplementedError
    else:
        X = adata.layers[layer]

    cells_selected = np.arange(X.shape[0])
    features_selected = np.arange(X.shape[1])

    n_cells = X.shape[0]
    feature_mean = (X.sum(0) / n_cells).A1
    feature_mean_sq = (X.multiply(X).sum(0) / n_cells).A1
    feature_var = (feature_mean_sq - feature_mean**2) * (
        n_cells / (n_cells - 1)
    )
    non_zero_features = (X > 0).sum(axis=0).A1
    pfeatures = np.isnan(non_zero_features) | (feature_var == 0)

    # cell x feature, filter feature
    X = X[:, ~pfeatures]
    features_selected = features_selected[~pfeatures]

    # gene x cell
    X = X.T
    X = X.multiply(1.0 / csr_matrix.sum(X, axis=0)) * 1e6
    X = X.tocsr()

    # filter cell, feature x cell
    feature_count_per_cell = (X > 0).sum(axis=0).A1
    pcells = np.isnan(feature_count_per_cell) | (
        feature_count_per_cell < cell_count
    )

    # feature x cell
    X = X[:, ~pcells]
    cells_selected = cells_selected[~pcells]
    feature_count_per_cell = feature_count_per_cell[~pcells]

    # census normalize, feature x cell
    census_X = (
        X.multiply(feature_count_per_cell)
        .multiply(1.0 / csr_matrix.sum(X, axis=0))
        .log1p()
        .tocsr()
    )

    # top variable features, feature x cell
    cell_count_per_feature = (census_X > 0).sum(axis=1).A1
    census_X_topcell = census_X[
        cell_count_per_feature >= 0.05 * census_X.shape[1], :
    ].tocsr()
    n_cells = census_X_topcell.shape[1]
    feature_mean = (census_X_topcell.sum(1) / n_cells).A1
    feature_mean_sq = (
        census_X_topcell.multiply(census_X_topcell).sum(1) / n_cells
    ).A1
    feature_var = (feature_mean_sq - feature_mean**2) * (
        n_cells / (n_cells - 1)
    )
    disp = feature_var / feature_mean

    # handle case less than 1000 features
    mvg = census_X_topcell[disp >= (np.sort(disp)[::-1])[:1000][-1], :]

    # top 1000 variable features for markov matrix
    selection = (mvg.sum(axis=0) != 0).A1
    mvg = mvg[:, selection]
    corr = compute_similarity1(mvg.A)
    markov = threshold_and_normalize_similarity_matrix(corr)

    # filter census output with nonzero cells in top variable features matrix
    cells_selected = cells_selected[selection]
    census_X = census_X[:, selection]
    feature_count_per_cell = feature_count_per_cell[selection]

    # calculate GC scores
    corrs = compute_similarity2(
        census_X.T.A, feature_count_per_cell.reshape(-1, 1)
    )[0, :]
    corrs[np.isnan(corrs)] = -1
    gcs = census_X[np.argsort(corrs)[::-1][:top_n_features], :].mean(axis=0).A1

    if not skip_regress:
        from scipy.optimize import nnls

        coef, err = nnls(markov, gcs)
        gcs = np.dot(markov, coef)

    gcs = diffused(markov, gcs)
    rank = rankdata(gcs)
    scores = rank / gcs.shape[0]

    adata.obs["gcs"] = np.nan
    adata.obs.iloc[cells_selected, adata.obs.columns.get_loc("gcs")] = gcs

    cytoGenes = compute_similarity2(census_X.T.A, scores.reshape(-1, 1))[0, :]

    adata.obs["cytotrace"] = np.nan
    adata.obs.iloc[
        cells_selected, adata.obs.columns.get_loc("cytotrace")
    ] = scores

    adata.obs["counts"] = np.nan
    adata.obs.iloc[cells_selected, adata.obs.columns.get_loc("counts")] = gcs

    adata.var["cytotrace"] = False
    adata.var.iloc[
        features_selected, adata.var.columns.get_loc("cytotrace")
    ] = True

    adata.var["cytotrace_corrs"] = np.nan
    adata.var.iloc[
        features_selected, adata.var.columns.get_loc("cytotrace_corrs")
    ] = np.array(corrs, dtype=np.float32)
    return {
        "CytoTRACE": scores,
        "CytoTRACErank": rank,
        "GCS": gcs,
        "Counts": feature_count_per_cell,
        "exprMatrix": census_X,
        "cytoGenes": cytoGenes,
        "gcsGenes": corrs,
        "filteredCells": np.setdiff1d(
            np.arange(adata.shape[0]), cells_selected
        ),
    }
