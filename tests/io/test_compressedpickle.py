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
