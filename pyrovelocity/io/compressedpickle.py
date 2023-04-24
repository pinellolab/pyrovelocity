import pickle
import zstandard as zstd
from typing import Any

class CompressedPickle:
    """
    A class for reading and writing zstandard-compressed pickle files.
    
    Examples:
    >>> import pandas as pd
    >>> test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> CompressedPickle.save('test_data.pkl.zst', test_data)
    >>> loaded_data = CompressedPickle.load('test_data.pkl.zst')
    >>> loaded_data.equals(test_data)
    True
    """
