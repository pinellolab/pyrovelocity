from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrovelocity.io.h5 import H5Accessor, load_from_h5, save_to_h5, hash_file


@pytest.fixture
def sample_data():
    return {
        "array": np.array([1, 2, 3, 4, 5]),
        "dataframe": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "list": ["a", "b", "c"],
        "nested": {
            "nested_array": np.array([6, 7, 8, 9, 10]),
            "nested_df": pd.DataFrame({"C": [7, 8, 9], "D": [10, 11, 12]}),
        },
    }


def test_save_and_load_h5(tmp_path, sample_data):
    file_path = tmp_path / "test.h5"

    saved_path, file_hash = save_to_h5(sample_data, file_path)
    assert saved_path == file_path
    assert isinstance(file_hash, str)
    assert file_path.exists()

    loaded_data = load_from_h5(file_path)
    assert isinstance(loaded_data, H5Accessor)

    np.testing.assert_array_equal(loaded_data.array[:], sample_data["array"])

    pd.testing.assert_frame_equal(
        loaded_data.dataframe[:], sample_data["dataframe"]
    )

    assert loaded_data.list == sample_data["list"]

    np.testing.assert_array_equal(
        loaded_data.nested["nested_array"][:],
        sample_data["nested"]["nested_array"],
    )
    pd.testing.assert_frame_equal(
        loaded_data.nested["nested_df"][:], sample_data["nested"]["nested_df"]
    )


def test_lazy_loading(tmp_path, sample_data):
    file_path = tmp_path / "test_lazy.h5"
    save_to_h5(sample_data, file_path)
    loaded_data = load_from_h5(file_path)

    assert isinstance(loaded_data.array, loaded_data.array.__class__)
    assert loaded_data.array.shape == sample_data["array"].shape
    assert loaded_data.array.dtype == sample_data["array"].dtype

    assert isinstance(loaded_data.dataframe, loaded_data.dataframe.__class__)
    assert loaded_data.dataframe.shape == sample_data["dataframe"].shape
    assert all(
        loaded_data.dataframe.columns == sample_data["dataframe"].columns
    )


def test_h5_accessor_methods(tmp_path, sample_data):
    file_path = tmp_path / "test_accessor.h5"
    save_to_h5(sample_data, file_path)
    accessor = load_from_h5(file_path)

    assert set(dir(accessor)) == set(sample_data.keys())

    assert repr(accessor) == f"H5Accessor(filename='{file_path}')"

    accessor.close()
    with pytest.raises(Exception):
        accessor.array[:]


def test_unsupported_type_warning(tmp_path):
    unsupported_data = {"unsupported": set([1, 2, 3])}
    file_path = tmp_path / "test_unsupported.h5"

    with pytest.warns(
        UserWarning, match="Skipping unsupported: unsupported type"
    ):
        save_to_h5(unsupported_data, file_path)


def test_nonexistent_attribute(tmp_path, sample_data):
    file_path = tmp_path / "test_nonexistent.h5"
    save_to_h5(sample_data, file_path)
    loaded_data = load_from_h5(file_path)

    with pytest.raises(AttributeError, match="No such attribute: nonexistent"):
        loaded_data.nonexistent


def test_dataframe_operations(tmp_path, sample_data):
    file_path = tmp_path / "test_df_ops.h5"
    save_to_h5(sample_data, file_path)
    loaded_data = load_from_h5(file_path)

    assert loaded_data.dataframe.head().equals(sample_data["dataframe"].head())

    np.testing.assert_array_equal(
        loaded_data.dataframe.A[:], sample_data["dataframe"]["A"]
    )


def test_file_hash_consistency(tmp_path, sample_data):
    file_path = tmp_path / "test_hash.h5"

    _, save_hash = save_to_h5(sample_data, file_path)

    load_from_h5(file_path)

    assert save_hash == hash_file(file_path)


@pytest.mark.parametrize("file_path", ["test.h5", Path("test.h5")])
def test_path_types(tmp_path, sample_data, file_path):
    full_path = tmp_path / file_path
    saved_path, _ = save_to_h5(sample_data, full_path)
    assert saved_path == full_path

    loaded_data = load_from_h5(full_path)
    assert isinstance(loaded_data, H5Accessor)
