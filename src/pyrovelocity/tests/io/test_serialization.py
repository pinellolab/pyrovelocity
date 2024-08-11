import tempfile

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from pyrovelocity.io.serialization import (
    create_sample_anndata,
    deserialize_anndata,
    load_anndata_from_json,
    save_anndata_to_json,
    serialize_anndata,
)


@pytest.fixture
def sample_adata():
    return create_sample_anndata(5, 4)


def test_create_sample_anndata():
    adata = create_sample_anndata(5, 4)
    assert isinstance(adata, AnnData)
    assert adata.n_obs == 5
    assert adata.n_vars == 4
    assert "batch" in adata.obs.columns
    assert "a" in adata.obsm
    assert "a" in adata.layers


def test_serialize_deserialize_anndata(sample_adata):
    serialized = serialize_anndata(sample_adata)
    deserialized = deserialize_anndata(serialized)

    assert np.allclose(sample_adata.X, deserialized.X)
    assert sample_adata.obs.equals(deserialized.obs)
    assert sample_adata.var.equals(deserialized.var)
    assert np.allclose(sample_adata.obsm["a"], deserialized.obsm["a"])
    assert np.allclose(sample_adata.layers["a"], deserialized.layers["a"])
    assert sample_adata.uns == deserialized.uns


def test_save_load_anndata_json(sample_adata, tmp_path):
    file_path = tmp_path / "test_adata.json"
    save_anndata_to_json(sample_adata, file_path)
    assert file_path.exists()

    loaded_adata = load_anndata_from_json(file_path)
    assert np.allclose(sample_adata.X, loaded_adata.X)
    assert sample_adata.obs.equals(loaded_adata.obs)
    assert sample_adata.var.equals(loaded_adata.var)
    assert np.allclose(sample_adata.obsm["a"], loaded_adata.obsm["a"])
    assert np.allclose(sample_adata.layers["a"], loaded_adata.layers["a"])
    assert sample_adata.uns == loaded_adata.uns


def test_serialize_large_anndata():
    large_adata = create_sample_anndata(1000, 100)
    serialized = serialize_anndata(large_adata)
    assert isinstance(serialized, dict)
    assert all(
        key in serialized
        for key in ["X", "obs", "var", "obsm", "layers", "uns"]
    )


def test_deserialize_invalid_data():
    invalid_data = {
        "X": [[1, 2], [3, 4]],
        # "obs": {},
        "var": {},
        "uns": {},
        "obsm": {},
        "varm": {},
        "layers": {},
        # "shape": (2, 2),
    }
    with pytest.raises(ValueError):
        deserialize_anndata(invalid_data)


def test_save_load_empty_anndata(tmp_path):
    empty_adata = AnnData(
        X=np.empty((0, 0)),
        obs=pd.DataFrame(index=[]),
        var=pd.DataFrame(index=[]),
    )
    file_path = tmp_path / "empty_adata.json"
    save_anndata_to_json(empty_adata, file_path)
    loaded_adata = load_anndata_from_json(file_path)
    assert loaded_adata.n_obs == 0
    assert loaded_adata.n_vars == 0


def test_serialize_deserialize_sparse_matrix():
    sparse_adata = AnnData(X=sparse.random(100, 100, density=0.1, format="csr"))
    sparse_adata.X[sparse_adata.X < 0.9] = 0
    sparse_adata.X = sparse_adata.X.tocsr()

    serialized = serialize_anndata(sparse_adata)
    deserialized = deserialize_anndata(serialized)

    assert np.allclose(sparse_adata.X.toarray(), deserialized.X)


def test_serialize_deserialize_with_complex_uns():
    adata = create_sample_anndata(100, 50)
    adata.uns["complex_item"] = {
        "nested": {"a": 1, "b": [1, 2, 3]},
        "array": np.array([1, 2, 3]),
    }

    serialized = serialize_anndata(adata)
    deserialized = deserialize_anndata(serialized)

    assert (
        deserialized.uns["complex_item"]["nested"]
        == adata.uns["complex_item"]["nested"]
    )
    assert np.array_equal(
        deserialized.uns["complex_item"]["array"],
        adata.uns["complex_item"]["array"],
    )


def test_load_anndata_from_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_anndata_from_json("nonexistent_file.json")


def test_save_anndata_to_invalid_path(sample_adata):
    with pytest.raises(OSError):
        save_anndata_to_json(sample_adata, "/invalid/path/adata.json")


def test_roundtrip_with_all_attributes():
    adata = create_sample_anndata(10, 5)
    adata.raw = adata.copy()
    adata.obsp["distances"] = np.random.random((10, 10))
    adata.varp["gene_correlation"] = np.random.random((5, 5))

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as tmp:
        save_anndata_to_json(adata, tmp.name)
        loaded_adata = load_anndata_from_json(tmp.name)

    assert np.allclose(adata.X, loaded_adata.X)
    assert adata.obs.equals(loaded_adata.obs)
    assert adata.var.equals(loaded_adata.var)
    assert np.allclose(adata.raw.X, loaded_adata.raw.X)
    assert np.allclose(adata.obsp["distances"], loaded_adata.obsp["distances"])
    assert np.allclose(
        adata.varp["gene_correlation"], loaded_adata.varp["gene_correlation"]
    )
