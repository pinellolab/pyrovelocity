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

    @staticmethod
    def save(file_path: str, obj: Any) -> None:
        """
        Save the given object to a zstandard-compressed pickle file.
        
        Args:
            file_path (str): The path of the file to save the object to.
            obj (object): The object to be saved.
            
        Examples:
        >>> import pandas as pd
        >>> test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> CompressedPickle.save('test_data.pkl.zst', test_data)
        """
        with open(file_path, 'wb') as f:
            compression_context = zstd.ZstdCompressor(level=3)
            with compression_context.stream_writer(f) as compressor:
                pickle.dump(obj, compressor)

