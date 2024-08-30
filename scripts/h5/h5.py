import weakref
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Dict, Tuple

from pyrovelocity.io.hash import hash_file
from pyrovelocity.logging import configure_logging

__all__ = ["save_to_h5", "load_from_h5"]

logger = configure_logging(__name__)


@beartype
def save_to_h5(
    data: Dict[str, Any],
    filename: str | Path,
) -> Tuple[Path, str]:
    with h5py.File(filename, "w") as f:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(
                    key,
                    data=value,
                    **hdf5plugin.Blosc2(
                        cname="zstd",
                        clevel=3,
                        filters=hdf5plugin.Blosc2.SHUFFLE,
                    ),
                )
            elif isinstance(value, pd.DataFrame):
                group = f.create_group(key)
                for column in value.columns:
                    group.create_dataset(
                        column,
                        data=value[column].values,
                        **hdf5plugin.Blosc2(
                            cname="zstd",
                            clevel=3,
                            filters=hdf5plugin.Blosc2.SHUFFLE,
                        ),
                    )
                group.attrs["columns"] = value.columns.tolist()
                group.attrs["index"] = value.index.tolist()
            elif isinstance(value, list):
                f.create_dataset(
                    key,
                    data=np.array(value, dtype=h5py.special_dtype(vlen=str)),
                    **hdf5plugin.Blosc2(
                        cname="zstd",
                        clevel=3,
                        filters=hdf5plugin.Blosc2.SHUFFLE,
                    ),
                )
            else:
                logger.warning(
                    f"Skipping {key}: unsupported type {type(value)}"
                )
    file_hash = _log_hash(filename, mode="saved")
    return Path(filename), file_hash


class LazyArray:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, key):
        return np.array(self.dataset[key])

    @property
    def shape(self):
        return self.dataset.shape

    @property
    def dtype(self):
        return self.dataset.dtype

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        return getattr(self.dataset, name)


class LazyDataFrame:
    def __init__(self, group):
        self.group = group
        self.columns = list(self.group.attrs["columns"])
        self.index = list(self.group.attrs["index"])

    def __getitem__(self, key):
        if isinstance(key, str):
            return LazyArray(self.group[key])
        else:
            df = pd.DataFrame(
                {col: self.group[col][()] for col in self.columns}
            )
            df.index = self.index
            return df[key]

    @property
    def shape(self):
        return (len(self.index), len(self.columns))

    def head(self, n=5):
        df = pd.DataFrame({col: self.group[col][:n] for col in self.columns})
        df.index = self.index[:n]
        return df

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        if name in self.columns:
            return LazyArray(self.group[name])
        return getattr(pd.DataFrame, name)


class H5Accessor:
    def __init__(self, filename):
        self._filename = filename
        self._file = None
        self._open_file()
        self._finalizer = weakref.finalize(self, self._close_file)

    def _open_file(self):
        if self._file is None or not self._file.id:
            self._file = h5py.File(self._filename, "r")

    def _close_file(self):
        if self._file is not None and self._file.id:
            self._file.close()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        try:
            self._open_file()
            if name not in self._file:
                raise AttributeError(f"No such attribute: {name}")

            item = self._file[name]
            if isinstance(item, h5py.Dataset):
                if item.dtype.kind == "S":
                    return item[()].astype(str).tolist()
                else:
                    return LazyArray(item)
            elif isinstance(item, h5py.Group):
                if "columns" in item.attrs:
                    return LazyDataFrame(item)
                else:
                    return {
                        key: self.__getattr__(f"{name}/{key}")
                        for key in item.keys()
                    }
            else:
                return item
        except Exception as e:
            if not name.startswith("_"):
                print(f"Error accessing {name}: {str(e)}")
            return None

    def __dir__(self):
        try:
            self._open_file()
            return list(self._file.keys())
        except Exception as e:
            print(f"Error listing attributes: {str(e)}")
            return []

    def close(self):
        self._close_file()

    def __repr__(self):
        return f"H5Accessor(filename='{self._filename}')"


@beartype
def load_from_h5(filename: str | Path) -> H5Accessor:
    accessor = H5Accessor(filename)
    _log_hash(filename, mode="loaded")
    return accessor


@beartype
def _log_hash(file_path: str | Path, mode: str = "loaded or saved") -> str:
    file_hash = hash_file(file_path=file_path)
    logger.info(
        f"\nSuccessfully {mode} file: {file_path}\n"
        f"SHA-256 hash: {file_hash}\n"
    )
    return file_hash
