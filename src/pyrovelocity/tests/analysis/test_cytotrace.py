"""Tests for `pyrovelocity.analysis.cytotrace` module."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix

from pyrovelocity.analysis import cytotrace


def test_load_cytotrace():
    print(cytotrace.__file__)


@pytest.fixture
def small_anndata():
    X = csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    obs = pd.DataFrame(index=["cell1", "cell2", "cell3"])
    var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
    adata = AnnData(X, obs=obs, var=var)
    adata.layers["raw"] = X
    return adata


def test_compute_similarity2():
    O = np.array([[1, 2, 3], [4, 5, 6]])
    P = np.array([[1, 2], [3, 4]])
    result = cytotrace.compute_similarity2(O, P)
    assert result.shape == (2, 3)
    assert np.allclose(result.T, np.corrcoef(O.T, P)[:3, 3:], atol=1e-5)


def test_compute_similarity1():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = cytotrace.compute_similarity1(A)
    assert result.shape == (3, 3)
    assert np.allclose(result, np.corrcoef(A.T))


def test_compute_gcs():
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    count = np.array([2, 3, 3])
    result = cytotrace.compute_gcs(mat, count, top_n_genes=2)
    assert result.shape == (3,)


def test_threshold_and_normalize_similarity_matrix():
    sim = np.array(
        [
            [1.0, 0.8, 0.3, 0.1],
            [0.8, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0],
        ]
    )

    result = cytotrace.threshold_and_normalize_similarity_matrix(sim)

    # check diagonal is zeroed out
    assert np.all(np.diag(result) == 0)

    # check values below or equal to mean are zeroed out
    mean_sim = np.mean(sim)
    assert np.all(result[sim <= mean_sim] == 0)

    # check non-zero rows are normalized to sum to 1
    non_zero_rows = np.where(result.sum(axis=1) > 0)[0]
    for row in non_zero_rows:
        assert_array_almost_equal(result[row].sum(), 1.0, decimal=6)

    # check zero rows remain zero
    zero_rows = np.where(result.sum(axis=1) == 0)[0]
    assert np.all(result[zero_rows] == 0)

    # check the result is sparse (contains zeros)
    assert np.sum(result == 0) > 0

    # check the result preserves symmetry
    if np.allclose(sim, sim.T):
        assert np.allclose(result, result.T)

    # check stronger similarities are preserved
    stronger_similarities = sim > np.mean(sim)
    assert np.all(result[stronger_similarities] >= 0)

    # check weaker similarities are removed
    weaker_similarities = sim <= np.mean(sim)
    assert np.all(result[weaker_similarities] == 0)

    # check behavior with all-zero input
    zero_sim = np.zeros_like(sim)
    zero_result = cytotrace.threshold_and_normalize_similarity_matrix(zero_sim)
    assert np.all(zero_result == 0)

    # check behavior with negative values
    neg_sim = np.array([[-1, 0.5], [0.5, -1]])
    neg_result = cytotrace.threshold_and_normalize_similarity_matrix(neg_sim)
    assert np.all(neg_result >= 0)
    assert_array_almost_equal(neg_result, np.array([[0, 1], [1, 0]]))


def test_diffused():
    markov = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]])
    gcs = np.array([1, 2, 3])
    result = cytotrace.diffused(markov, gcs)
    assert result.shape == gcs.shape


def test_cytotrace_sparse(small_anndata):
    result = cytotrace.cytotrace_sparse(small_anndata, layer="raw")

    assert isinstance(result, dict)
    assert "CytoTRACE" in result
    assert "GCS" in result
    assert "cytoGenes" in result

    assert "gcs" in small_anndata.obs.columns
    assert "cytotrace" in small_anndata.obs.columns
    assert "counts" in small_anndata.obs.columns
    assert "cytotrace" in small_anndata.var.columns
    assert "cytotrace_corrs" in small_anndata.var.columns


def test_cytotrace_sparse_skipregress(small_anndata):
    result = cytotrace.cytotrace_sparse(
        small_anndata, layer="raw", skip_regress=True
    )

    assert isinstance(result, dict)
    assert "CytoTRACE" in result
    assert "GCS" in result
    assert "cytoGenes" in result

    assert "gcs" in small_anndata.obs.columns
    assert "cytotrace" in small_anndata.obs.columns
    assert "counts" in small_anndata.obs.columns
    assert "cytotrace" in small_anndata.var.columns
    assert "cytotrace_corrs" in small_anndata.var.columns


def test_cytotrace_sparse_errors():
    adata = AnnData(X=np.array([[1, 2], [3, 4]]))
    adata.layers["raw"] = adata.X

    with pytest.raises(
        NotImplementedError,
    ):
        cytotrace.cytotrace_sparse(adata)

    with pytest.raises(
        KeyError,
    ):
        cytotrace.cytotrace_sparse(adata, layer="non_existent")
