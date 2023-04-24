import pytest
import os
import pandas as pd
import numpy as np
from pyrovelocity.io.compressedpickle import CompressedPickle

@pytest.fixture(scope="module")
def test_data():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

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
    test_dict = {'a': np.arange(5), 'b': np.linspace(0, 1, 5)}
    CompressedPickle.save(test_file, test_dict)
    loaded_dict = CompressedPickle.load(test_file)
    assert compare_dicts(test_dict, loaded_dict)


def test_load_non_existent_file():
    with pytest.raises(FileNotFoundError):
        CompressedPickle.load('non_existent.pkl.zst')

