import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from pyrovelocity.io.hash import hash_file
from pyrovelocity.io.serialization import (
    create_sample_anndata,
    deserialize_anndata,
    load_anndata_from_json,
    save_anndata_to_json,
    serialize_anndata,
)


@pytest.fixture(scope="function")
def sample_adata():
    """Create a sample AnnData object for testing."""
    return create_sample_anndata(5, 4)


@pytest.fixture(scope="function")
def mock_hash_file():
    """Mock the hash_file function."""
    with patch("pyrovelocity.io.serialization.hash_file") as mock:
        mock.return_value = "mocked_hash"
        yield mock


def test_create_sample_anndata():
    """Test the creation of a sample AnnData object."""
    adata = create_sample_anndata(5, 4)
    assert isinstance(adata, AnnData), "Expected an AnnData object"
    assert adata.n_obs == 5, "Expected 5 observations"
    assert adata.n_vars == 4, "Expected 4 variables"
    assert "batch" in adata.obs.columns, "Expected 'batch' column in obs"
    assert "a" in adata.obsm, "Expected 'a' in obsm"
    assert "a" in adata.layers, "Expected 'a' in layers"


def test_serialize_deserialize_anndata(sample_adata):
    """Test serialization and deserialization of AnnData object."""
    serialized = serialize_anndata(sample_adata)
    deserialized = deserialize_anndata(serialized)

    assert np.allclose(sample_adata.X, deserialized.X), "Mismatch in X matrix"
    assert sample_adata.obs.equals(
        deserialized.obs
    ), "Mismatch in obs DataFrame"
    assert sample_adata.var.equals(
        deserialized.var
    ), "Mismatch in var DataFrame"
    assert np.allclose(
        sample_adata.obsm["a"], deserialized.obsm["a"]
    ), "Mismatch in obsm['a']"
    assert np.allclose(
        sample_adata.layers["a"], deserialized.layers["a"]
    ), "Mismatch in layers['a']"
    assert sample_adata.uns == deserialized.uns, "Mismatch in uns dictionary"


def test_save_load_anndata_json(sample_adata, tmp_path, mock_hash_file, caplog):
    """Test saving and loading AnnData object to/from JSON."""
    file_path = tmp_path / "test_adata.json"

    save_anndata_to_json(sample_adata, file_path)
    assert (
        file_path.exists()
    ), f"Expected file {file_path} to exist after saving"
    mock_hash_file.assert_called_once_with(file_path)
    assert (
        f"\nSaved file: {file_path}\nSHA-256 hash: mocked_hash\n" in caplog.text
    )

    caplog.clear()

    loaded_adata = load_anndata_from_json(file_path)
    mock_hash_file.assert_called_with(file_path)
    assert (
        f"\nLoading file: {file_path}\nSHA-256 hash: mocked_hash\n"
        in caplog.text
    )

    assert np.allclose(
        sample_adata.X, loaded_adata.X
    ), "Mismatch in X matrix after loading"
    assert sample_adata.obs.equals(
        loaded_adata.obs
    ), "Mismatch in obs DataFrame after loading"
    assert sample_adata.var.equals(
        loaded_adata.var
    ), "Mismatch in var DataFrame after loading"
    assert np.allclose(
        sample_adata.obsm["a"], loaded_adata.obsm["a"]
    ), "Mismatch in obsm['a'] after loading"
    assert np.allclose(
        sample_adata.layers["a"], loaded_adata.layers["a"]
    ), "Mismatch in layers['a'] after loading"
    assert (
        sample_adata.uns == loaded_adata.uns
    ), "Mismatch in uns dictionary after loading"


@pytest.fixture
def expected_hash(sample_adata, tmp_path):
    """Generate expected hash for a sample AnnData object."""
    file_path = tmp_path / "temp_adata.json"
    save_anndata_to_json(sample_adata, file_path)
    return hash_file(file_path)


def test_save_load_anndata_json_with_hash(
    sample_adata, tmp_path, expected_hash, caplog
):
    """Test saving and loading AnnData object to/from JSON with hash validation."""
    file_path = tmp_path / "test_adata.json"

    saved_hash = save_anndata_to_json(
        sample_adata, file_path, expected_hash=expected_hash
    )
    assert saved_hash == expected_hash
    assert "Hash validation succeeded" in caplog.text

    caplog.clear()

    loaded_adata = load_anndata_from_json(
        file_path, expected_hash=expected_hash
    )
    assert "Hash validation succeeded" in caplog.text

    assert np.allclose(sample_adata.X, loaded_adata.X)
    assert sample_adata.obs.equals(loaded_adata.obs)
    assert sample_adata.var.equals(loaded_adata.var)


def test_save_load_anndata_json_with_incorrect_hash(
    sample_adata, tmp_path, expected_hash, caplog
):
    """Test saving and loading AnnData object to/from JSON with incorrect hash."""
    file_path = tmp_path / "test_adata.json"
    incorrect_hash = "incorrect_hash_value"
    assert incorrect_hash != expected_hash

    save_anndata_to_json(sample_adata, file_path, expected_hash=incorrect_hash)
    assert "Hash mismatch" in caplog.text

    caplog.clear()

    load_anndata_from_json(file_path, expected_hash=incorrect_hash)
    assert "Hash mismatch" in caplog.text


def test_serialize_large_anndata():
    """Test serialization of a large AnnData object."""
    large_adata = create_sample_anndata(1000, 100)
    serialized = serialize_anndata(large_adata)
    assert isinstance(
        serialized, dict
    ), "Expected serialized object to be a dictionary"
    assert all(
        key in serialized
        for key in ["X", "obs", "var", "obsm", "layers", "uns"]
    ), "Missing expected keys in serialized dictionary"


def test_deserialize_invalid_data():
    """Test deserialization with invalid data."""
    invalid_data = {
        "X": [[1, 2], [3, 4]],
        "var": {},
        "uns": {},
        "obsm": {},
        "varm": {},
        "layers": {},
    }
    with pytest.raises(
        ValueError, match="Invalid data format: missing required keys"
    ):
        deserialize_anndata(invalid_data)


def test_save_load_empty_anndata(tmp_path, mock_hash_file, caplog):
    """Test saving and loading an empty AnnData object."""
    empty_adata = AnnData(
        X=np.empty((0, 0)),
        obs=pd.DataFrame(index=[]),
        var=pd.DataFrame(index=[]),
    )
    file_path = tmp_path / "empty_adata.json"
    save_anndata_to_json(empty_adata, file_path)
    loaded_adata = load_anndata_from_json(file_path)
    assert (
        loaded_adata.n_obs == 0
    ), "Expected 0 observations in loaded empty AnnData"
    assert (
        loaded_adata.n_vars == 0
    ), "Expected 0 variables in loaded empty AnnData"
    assert (
        mock_hash_file.call_count == 2
    ), "Expected hash_file to be called twice"


def test_serialize_deserialize_sparse_matrix():
    """Test serialization and deserialization of AnnData with sparse matrix."""
    sparse_adata = AnnData(X=sparse.random(100, 100, density=0.1, format="csr"))
    sparse_adata.X[sparse_adata.X < 0.9] = 0
    sparse_adata.X = sparse_adata.X.tocsr()

    serialized = serialize_anndata(sparse_adata)
    deserialized = deserialize_anndata(serialized)

    assert np.allclose(
        sparse_adata.X.toarray(), deserialized.X
    ), "Mismatch in sparse matrix after serialization/deserialization"


def test_serialize_deserialize_with_complex_uns():
    """Test serialization and deserialization of AnnData with complex uns."""
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
    ), "Mismatch in nested dictionary in uns"
    assert np.array_equal(
        deserialized.uns["complex_item"]["array"],
        adata.uns["complex_item"]["array"],
    ), "Mismatch in numpy array in uns"


def test_serialize_deserialize_with_nested_numpy_array_in_uns(sample_adata):
    """Test serialization and deserialization of AnnData with nested numpy arrays in uns."""
    adata = sample_adata.copy()
    adata.uns["numpy_array"] = np.array([1, 2, 3])
    adata.uns["nested_dict"] = {"array": np.array([4, 5, 6])}

    serialized = serialize_anndata(adata)
    deserialized = deserialize_anndata(serialized)

    assert isinstance(
        deserialized.uns["numpy_array"], np.ndarray
    ), "Expected numpy array in uns"
    assert np.array_equal(
        adata.uns["numpy_array"], deserialized.uns["numpy_array"]
    ), "Mismatch in numpy array in uns"
    assert isinstance(
        deserialized.uns["nested_dict"]["array"], np.ndarray
    ), "Expected numpy array in nested dict in uns"
    assert np.array_equal(
        adata.uns["nested_dict"]["array"],
        deserialized.uns["nested_dict"]["array"],
    ), "Mismatch in numpy array in nested dict in uns"


def test_load_anndata_from_nonexistent_file():
    """Test loading AnnData from a nonexistent file."""
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        load_anndata_from_json("nonexistent_file.json")


def test_save_anndata_to_invalid_path(sample_adata):
    """Test saving AnnData to an invalid path."""
    with pytest.raises(OSError):
        save_anndata_to_json(sample_adata, "/invalid/path/adata.json")


def test_roundtrip_with_all_attributes(tmp_path, mock_hash_file):
    """Test full roundtrip serialization and deserialization with all AnnData attributes."""
    adata = create_sample_anndata(10, 5)
    adata.raw = adata.copy()
    adata.obsp["distances"] = np.random.random((10, 10))
    adata.varp["gene_correlation"] = np.random.random((5, 5))

    file_path = tmp_path / "roundtrip_adata.json"
    save_anndata_to_json(adata, file_path)
    loaded_adata = load_anndata_from_json(file_path)

    assert np.allclose(
        adata.X, loaded_adata.X
    ), "Mismatch in X matrix after roundtrip"
    assert adata.obs.equals(
        loaded_adata.obs
    ), "Mismatch in obs DataFrame after roundtrip"
    assert adata.var.equals(
        loaded_adata.var
    ), "Mismatch in var DataFrame after roundtrip"
    assert np.allclose(
        adata.raw.X, loaded_adata.raw.X
    ), "Mismatch in raw.X after roundtrip"
    assert np.allclose(
        adata.obsp["distances"], loaded_adata.obsp["distances"]
    ), "Mismatch in obsp['distances'] after roundtrip"
    assert np.allclose(
        adata.varp["gene_correlation"], loaded_adata.varp["gene_correlation"]
    ), "Mismatch in varp['gene_correlation'] after roundtrip"
    assert (
        mock_hash_file.call_count == 2
    ), "Expected hash_file to be called twice"


def test_deserialize_with_sparse_layers(sample_adata, tmp_path):
    """Test deserialization with sparse layers."""
    file_path = tmp_path / "sparse_layers_test.json"
    save_anndata_to_json(sample_adata, file_path)

    sparse_loaded_adata = load_anndata_from_json(file_path, sparse_layers=True)

    for layer_key in sparse_loaded_adata.layers:
        assert sparse.issparse(
            sparse_loaded_adata.layers[layer_key]
        ), f"Expected layer '{layer_key}' to be sparse when sparse_layers=True"

    for layer_key in sample_adata.layers:
        assert np.allclose(
            sample_adata.layers[layer_key],
            sparse_loaded_adata.layers[layer_key].toarray(),
        ), f"Mismatch in layer '{layer_key}' data when using sparse_layers=True"

    dense_loaded_adata = load_anndata_from_json(file_path)

    for layer_key in dense_loaded_adata.layers:
        assert not sparse.issparse(
            dense_loaded_adata.layers[layer_key]
        ), f"Expected layer '{layer_key}' to be dense when sparse_layers=False"


def test_load_anndata_from_json_passes_sparse_layers(sample_adata, tmp_path):
    """Test that load_anndata_from_json correctly passes sparse_layers to deserialize_anndata."""
    file_path = tmp_path / "sparse_param_test.json"
    save_anndata_to_json(sample_adata, file_path)

    with patch(
        "pyrovelocity.io.serialization.deserialize_anndata"
    ) as mock_deserialize:
        mock_deserialize.return_value = sample_adata.copy()

        load_anndata_from_json(file_path, sparse_layers=True)
        mock_deserialize.assert_called_once()
        assert mock_deserialize.call_args[1]["sparse_layers"] is True

        mock_deserialize.reset_mock()

        load_anndata_from_json(file_path)
        mock_deserialize.assert_called_once()
        assert mock_deserialize.call_args[1]["sparse_layers"] is False


def test_deserialize_anndata_sparse_layers_direct():
    """Test that deserialize_anndata correctly handles the sparse_layers parameter directly."""
    data = {
        "shape": [5, 4],
        "obs": {"index": ["cell1", "cell2", "cell3", "cell4", "cell5"]},
        "var": {"index": ["gene1", "gene2", "gene3", "gene4"]},
        "X": [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ],
        "layers": {
            "spliced": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ],
            "unspliced": [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
                [1.7, 1.8, 1.9, 2.0],
            ],
        },
    }

    sparse_adata = deserialize_anndata(data, sparse_layers=True)

    for layer_key in sparse_adata.layers:
        assert sparse.issparse(
            sparse_adata.layers[layer_key]
        ), f"Expected layer '{layer_key}' to be sparse when sparse_layers=True"

    dense_adata = deserialize_anndata(data, sparse_layers=False)

    for layer_key in dense_adata.layers:
        assert not sparse.issparse(
            dense_adata.layers[layer_key]
        ), f"Expected layer '{layer_key}' to be dense when sparse_layers=False"

    for layer_key in sparse_adata.layers:
        sparse_data = (
            sparse_adata.layers[layer_key].toarray()
            if sparse.issparse(sparse_adata.layers[layer_key])
            else sparse_adata.layers[layer_key]
        )
        dense_data = dense_adata.layers[layer_key]
        assert np.allclose(
            sparse_data, dense_data
        ), f"Data mismatch in layer '{layer_key}' between sparse and dense formats"


def test_handle_corrupted_json(tmp_path):
    """Test handling of corrupted JSON files."""
    file_path = tmp_path / "corrupted.json"
    with file_path.open("w") as f:
        f.write("{This is not valid JSON")

    with pytest.raises(json.JSONDecodeError):
        load_anndata_from_json(file_path)
