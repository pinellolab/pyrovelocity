import anndata
import numpy as np
import pytest
import scvelo as scv
from pyrovelocity.metrics.criticality_index import (
    calculate_criticality_index,
)


@pytest.fixture(scope="module")
def adata_fixture():
    return scv.datasets.simulation(
        random_seed=0,
        n_obs=10,
        n_vars=3,
        alpha=5,
        beta=0.5,
        gamma=0.3,
        alpha_=0,
        switches=[1, 5, 10],
        noise_model="gillespie",
    )


def test_calculate_criticality_index(adata_fixture):
    criticality_index, _, _, _ = calculate_criticality_index(adata_fixture)
    assert isinstance(criticality_index, float)
    assert 0 <= criticality_index <= 1


def test_empty_anndata():
    empty_adata = anndata.AnnData(X=np.empty((0, 0)))
    with pytest.raises(ValueError):
        calculate_criticality_index(empty_adata)


def test_missing_unspliced_layer(adata_fixture):
    adata_no_unspliced = adata_fixture.copy()
    del adata_no_unspliced.layers["unspliced"]
    with pytest.raises(KeyError):
        calculate_criticality_index(adata_no_unspliced)


def test_missing_spliced_layer(adata_fixture):
    adata_no_spliced = adata_fixture.copy()
    del adata_no_spliced.layers["spliced"]
    with pytest.raises(KeyError):
        calculate_criticality_index(adata_no_spliced)
