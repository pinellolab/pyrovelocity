import os
from os import PathLike
from pathlib import Path

import dill as pickle
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict
from sparse import COO
from zstandard import (
    ZstdCompressionParameters,
    ZstdCompressor,
    ZstdDecompressor,
)

from pyrovelocity.io.hash import hash_file
from pyrovelocity.io.sparsity import densify_arrays, sparsify_arrays
from pyrovelocity.logging import configure_logging

__all__ = ["CompressedPickle"]

logger = configure_logging(__name__)


@beartype
def get_cpu_count() -> int:
    """
    Safely determine the number of CPUs in the system.
    Falls back to a default value if it can't be determined.
    """
    try:
        return os.cpu_count() or 1
    except NotImplementedError:
        return 1


@beartype
def get_optimal_thread_count(cpu_count: int) -> int:
    """
    Determine the optimal number of threads based on CPU count.
    """
    if cpu_count <= 2:
        return cpu_count
    elif cpu_count <= 8:
        return cpu_count - 1
    else:
        return cpu_count - 2


CPU_COUNT = get_cpu_count()
COMPRESSION_THREADS = get_optimal_thread_count(CPU_COUNT)


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
        density_threshold: float = 0.3,
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
                obj = sparsify_arrays(
                    data_dict=obj,
                    density_threshold=density_threshold,
                )
            else:
                logger.debug(
                    """
                    The object is not a dictionary of numpy arrays or COO objects.
                    It cannot be automatically sparsified.
                    """
                )

        compression_params = ZstdCompressionParameters(
            compression_level=3,
            threads=COMPRESSION_THREADS,
        )

        with file_path.open("wb") as f:
            compression_context = ZstdCompressor(
                compression_params=compression_params
            )
            with compression_context.stream_writer(f) as compressor:
                pickle.dump(obj, compressor)

        _log_hash(file_path=file_path, mode="saved")
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
            decompression_context = ZstdDecompressor()
            with decompression_context.stream_reader(f) as decompressor:
                obj = pickle.load(decompressor)

        if densify:
            if isinstance(obj, Dict) and all(
                isinstance(v, (np.ndarray, COO)) for v in obj.values()
            ):
                obj = densify_arrays(obj)
            else:
                logger.debug(
                    """
                    The object is not a dictionary of numpy arrays or COO objects.
                    It cannot be automatically densified.
                    """
                )
        _log_hash(file_path=file_path, mode="loaded")
        return obj


@beartype
def _log_hash(file_path: str | Path, mode: str = "loaded or saved") -> str:
    file_hash = hash_file(file_path=file_path)
    logger.info(
        f"\nSuccessfully {mode} file: {file_path}\n"
        f"SHA-256 hash: {file_hash}\n"
    )
    return file_hash
