import os

import numpy as np
import pandas as pd
import pytest

from pyrovelocity.io.compressedpickle import CompressedPickle


@pytest.fixture(scope="module")
def test_data():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture(scope="module")
def test_file():
    return "test_data.pkl.zst"


def test_save(test_data, test_file):
    CompressedPickle.save(test_file, test_data)
    assert os.path.exists(test_file)


def test_load(test_data, test_file):
    loaded_data = CompressedPickle.load(test_file)
    assert loaded_data.equals(test_data)


def compare_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        if isinstance(dict1[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                return False
        elif dict1[key] != dict2[key]:
            return False

    return True


def test_save_load_dict(test_file):
    test_dict = {"a": np.arange(5), "b": np.linspace(0, 1, 5)}
    CompressedPickle.save(test_file, test_dict)
    loaded_dict = CompressedPickle.load(test_file)
    assert compare_dicts(test_dict, loaded_dict)


def compare_complex_objects(obj1, obj2):
    if len(obj1) != len(obj2):
        return False

    for item1, item2 in zip(obj1, obj2):
        if isinstance(item1, dict) and isinstance(item2, dict):
            if not compare_dicts(item1, item2):
                return False
        elif isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            if not np.array_equal(item1, item2):
                return False
        elif item1 != item2:
            return False

    return True


def test_save_load_complex_object(test_file):
    test_obj = [{"a": np.array([1, 2, 3]), "b": 1.5}, (4, "test", [1, 2])]
    CompressedPickle.save(test_file, test_obj)
    loaded_obj = CompressedPickle.load(test_file)
    assert compare_complex_objects(test_obj, loaded_obj)


def test_load_non_existent_file():
    with pytest.raises(FileNotFoundError):
        CompressedPickle.load("non_existent.pkl.zst")


@pytest.fixture(scope="module", autouse=True)
def cleanup(request, test_file):
    def delete_test_file():
        if os.path.exists(test_file):
            os.remove(test_file)

    request.addfinalizer(delete_test_file)
