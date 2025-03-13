import os

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sparse import COO

from pyrovelocity.io.compressedpickle import CompressedPickle


@pytest.fixture
def test_dataframe():
    """Create a simple pandas DataFrame for testing."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def test_dict():
    """Create a dictionary with numpy arrays for testing."""
    return {"a": np.arange(5), "b": np.linspace(0, 1, 5)}


@pytest.fixture
def test_complex_object():
    """Create a complex nested object for testing."""
    return [{"a": np.array([1, 2, 3]), "b": 1.5}, (4, "test", [1, 2])]


@pytest.fixture
def sparse_dict():
    """Create a dictionary with arrays of varying sparsity for testing."""
    return {
        "dense": np.array([[1, 2], [3, 4]]),
        "sparse": np.array([[1, 0], [0, 2]]),
        "very_sparse": np.array([[0, 1], [0, 0]]),
    }


@pytest.fixture
def mixed_dict():
    """Create a dictionary with both numpy arrays and sparse arrays."""
    return {
        "dense": np.array([[1, 2], [3, 4]]),
        "sparse": COO.from_numpy(np.array([[1, 0], [0, 2]])),
        "very_sparse": COO.from_numpy(np.array([[0, 1], [0, 0]])),
        "non_array": "test string",
    }


def test_save_and_load_dataframe(
    test_dataframe, temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving and loading a pandas DataFrame."""
    loaded_data = save_and_load_helper(
        test_dataframe, temp_compressed_pickle_path
    )
    assert loaded_data.equals(test_dataframe)


def test_save_and_load_dict(
    test_dict, temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving and loading a dictionary with numpy arrays."""
    loaded_dict = save_and_load_helper(test_dict, temp_compressed_pickle_path)

    # Verify keys match
    assert loaded_dict.keys() == test_dict.keys()

    # Verify array contents match
    for key in test_dict:
        assert np.array_equal(loaded_dict[key], test_dict[key])


def test_save_and_load_complex_object(
    test_complex_object, temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving and loading a complex nested object."""
    loaded_obj = save_and_load_helper(
        test_complex_object, temp_compressed_pickle_path
    )

    # Verify structure
    assert len(loaded_obj) == len(test_complex_object)

    # Verify first item (dictionary)
    assert loaded_obj[0].keys() == test_complex_object[0].keys()
    assert np.array_equal(loaded_obj[0]["a"], test_complex_object[0]["a"])
    assert loaded_obj[0]["b"] == test_complex_object[0]["b"]

    # Verify second item (tuple)
    assert loaded_obj[1] == test_complex_object[1]


def test_load_non_existent_file():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        CompressedPickle.load("non_existent_file.pkl.zst")


def test_save_and_load_sparse_dict(
    sparse_dict, temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving and loading a dictionary with arrays of varying sparsity."""
    # Use sparsification with a threshold that will convert some arrays to sparse format
    loaded_dict = save_and_load_helper(
        sparse_dict, temp_compressed_pickle_path, density_threshold=0.5
    )

    # Verify all arrays are loaded as dense numpy arrays by default
    assert isinstance(loaded_dict["dense"], np.ndarray)
    assert isinstance(loaded_dict["sparse"], np.ndarray)
    assert isinstance(loaded_dict["very_sparse"], np.ndarray)

    # Verify contents match
    assert np.array_equal(loaded_dict["dense"], sparse_dict["dense"])
    assert np.array_equal(loaded_dict["sparse"], sparse_dict["sparse"])
    assert np.array_equal(
        loaded_dict["very_sparse"], sparse_dict["very_sparse"]
    )


def test_save_and_load_sparse_dict_without_densify(
    sparse_dict, temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving and loading a dictionary with arrays, keeping sparse arrays sparse."""
    # Use sparsification with a threshold and load without densifying
    loaded_dict = save_and_load_helper(
        sparse_dict,
        temp_compressed_pickle_path,
        density_threshold=0.5,
        densify=False,
    )

    # Verify dense arrays remain dense, sparse arrays are loaded as sparse
    assert isinstance(loaded_dict["dense"], np.ndarray)
    assert isinstance(loaded_dict["sparse"], COO)
    assert isinstance(loaded_dict["very_sparse"], COO)

    # Verify contents match
    assert np.array_equal(loaded_dict["dense"], sparse_dict["dense"])
    assert np.array_equal(
        loaded_dict["sparse"].todense(), sparse_dict["sparse"]
    )
    assert np.array_equal(
        loaded_dict["very_sparse"].todense(), sparse_dict["very_sparse"]
    )


def test_save_without_sparsify(
    sparse_dict, temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving without sparsification."""
    loaded_dict = save_and_load_helper(
        sparse_dict, temp_compressed_pickle_path, sparsify=False
    )

    # Verify all arrays remain dense
    assert all(isinstance(arr, np.ndarray) for arr in loaded_dict.values())

    # Verify contents match
    assert all(
        np.array_equal(loaded_dict[key], sparse_dict[key])
        for key in sparse_dict
    )


def test_save_and_load_mixed_dict(
    mixed_dict, temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving and loading a dictionary with both numpy and sparse arrays."""
    loaded_dict = save_and_load_helper(mixed_dict, temp_compressed_pickle_path)

    # Verify types
    assert isinstance(loaded_dict["dense"], np.ndarray)
    assert isinstance(loaded_dict["sparse"], COO)
    assert isinstance(loaded_dict["very_sparse"], COO)
    assert isinstance(loaded_dict["non_array"], str)

    # Verify contents
    assert np.array_equal(loaded_dict["dense"], mixed_dict["dense"])
    assert np.array_equal(
        loaded_dict["sparse"].todense(), mixed_dict["sparse"].todense()
    )
    assert np.array_equal(
        loaded_dict["very_sparse"].todense(),
        mixed_dict["very_sparse"].todense(),
    )
    assert loaded_dict["non_array"] == mixed_dict["non_array"]


def test_save_and_load_non_dict(
    temp_compressed_pickle_path, save_and_load_helper
):
    """Test saving and loading a non-dictionary object."""
    non_dict_obj = np.array([[1, 2], [3, 4]])
    loaded_obj = save_and_load_helper(non_dict_obj, temp_compressed_pickle_path)

    assert isinstance(loaded_obj, np.ndarray)
    assert np.array_equal(loaded_obj, non_dict_obj)


def test_file_exists_after_save(test_dataframe, temp_compressed_pickle_path):
    """Test that the file exists after saving."""
    CompressedPickle.save(temp_compressed_pickle_path, test_dataframe)
    assert os.path.exists(temp_compressed_pickle_path)
