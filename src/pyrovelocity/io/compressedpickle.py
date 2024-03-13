import os
import pickle
from pathlib import Path
from typing import Any

import zstandard as zstd


__all__ = ["CompressedPickle"]


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
    def save(file_path: os.PathLike | str, obj: Any) -> None:
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
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as f:
            compression_context = zstd.ZstdCompressor(level=3)
            with compression_context.stream_writer(f) as compressor:
                pickle.dump(obj, compressor)

    @staticmethod
    def load(file_path: os.PathLike | str) -> Any:
        """
        Load an object from a zstandard-compressed pickle file.

        Args:
            file_path (str): The path of the file to load the object from.

        Returns:
            object: The loaded object.

        Examples:
        >>> import pandas as pd
        >>> test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> CompressedPickle.save('test_data.pkl.zst', test_data)
        >>> loaded_data = CompressedPickle.load('test_data.pkl.zst')
        >>> loaded_data.equals(test_data)
        True
        """
        with open(file_path, "rb") as f:
            decompression_context = zstd.ZstdDecompressor()
            with decompression_context.stream_reader(f) as decompressor:
                obj = pickle.load(decompressor)
        return obj
