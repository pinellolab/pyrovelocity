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

