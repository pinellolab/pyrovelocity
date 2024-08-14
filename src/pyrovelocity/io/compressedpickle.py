import pickle
from os import PathLike
from pathlib import Path

import numpy as np
import zstandard as zstd
from beartype.typing import Any, Dict
from sparse import COO

from pyrovelocity.io.sparsity import densify_arrays, sparsify_arrays
from pyrovelocity.logging import configure_logging

__all__ = ["CompressedPickle"]

logger = configure_logging(__name__)


# TODO: Handle sparsification when values are not exclusively arrays
class CompressedPickle:
    """
    A class for reading and writing zstandard-compressed pickle files.

    Examples:
    >>> import pandas as pd
    >>> tmp = getfixture("tmp_path")
    >>> test_data_path = tmp / "test_data.pkl.zst"
    >>> test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> CompressedPickle.save(test_data_path, test_data)
    >>> loaded_data = CompressedPickle.load(test_data_path)
    >>> loaded_data.equals(test_data)
    True
    """

    @staticmethod
    def save(
        file_path: PathLike | str,
        obj: Any,
        sparsify: bool = True,
    ) -> Path:
        """
        Save the given object to a zstandard-compressed pickle file.

        Args:
            file_path (str): The path of the file to save the object to.
            obj (object): The object to be saved.

        Examples:
        >>> import pandas as pd
        >>> tmp = getfixture("tmp_path")
        >>> test_data_path = tmp / "test_data.pkl.zst"
        >>> test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> CompressedPickle.save(test_data_path, test_data)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if sparsify:
            if isinstance(obj, Dict) and all(
                isinstance(v, (np.ndarray, COO)) for v in obj.values()
            ):
                obj = sparsify_arrays(obj)
            else:
                logger.warning(
                    """
                    The object is not a dictionary of numpy arrays or COO objects.
                    It cannot be automatically sparsified.
                    """
                )

        with file_path.open("wb") as f:
            compression_context = zstd.ZstdCompressor(level=3)
            with compression_context.stream_writer(f) as compressor:
                pickle.dump(obj, compressor)

        return file_path

    @staticmethod
    def load(
        file_path: PathLike | str,
        densify: bool = True,
    ) -> Any:
        """
        Load an object from a zstandard-compressed pickle file.

        Args:
            file_path (str): The path of the file to load the object from.

        Returns:
            object: The loaded object.

        Examples:
        >>> import pandas as pd
        >>> tmp = getfixture("tmp_path")
        >>> test_data_path = tmp / "test_data.pkl.zst"
        >>> test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> CompressedPickle.save(test_data_path, test_data)
        >>> loaded_data = CompressedPickle.load(test_data_path)
        >>> loaded_data.equals(test_data)
        True
        """
        with open(file_path, "rb") as f:
            decompression_context = zstd.ZstdDecompressor()
            with decompression_context.stream_reader(f) as decompressor:
                obj = pickle.load(decompressor)

        if densify:
            if isinstance(obj, Dict) and all(
                isinstance(v, (np.ndarray, COO)) for v in obj.values()
            ):
                obj = densify_arrays(obj)
            else:
                logger.warning(
                    """
                    The object is not a dictionary of numpy arrays or COO objects.
                    It cannot be automatically densified.
                    """
                )
        return obj
