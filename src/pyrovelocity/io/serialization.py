import json
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata import Raw as AnnDataRaw
from beartype import beartype
from beartype.typing import Any, Dict
from scipy import sparse

from pyrovelocity.io.hash import hash_file
from pyrovelocity.utils import (
    configure_logging,
    ensure_numpy_array,
    pretty_log_dict,
    pretty_print_dict,
)

__all__ = [
    "serialize_anndata",
    "deserialize_anndata",
    "save_anndata_to_json",
    "load_anndata_from_json",
    "create_sample_anndata",
]

logger = configure_logging(__name__)


@beartype
def serialize_anndata(adata: AnnData | AnnDataRaw) -> Dict[str, Any]:
    """
    Serialize an AnnData object to a dictionary.

    Args:
        adata: AnnData object to serialize

    Returns:
        Dictionary representation of the AnnData object
    """
    if isinstance(adata, AnnDataRaw):
        adata = adata.to_adata()

    serialized = {
        "shape": adata.shape,
        "obs": adata.obs.reset_index().to_dict(orient="list"),
        "var": adata.var.reset_index().to_dict(orient="list"),
    }

    if adata.X is not None:
        X = ensure_numpy_array(adata.X)
        serialized["X"] = X.tolist() if X.size > 0 else [[], []]

    if adata.layers:
        serialized["layers"] = {
            key: ensure_numpy_array(value)
            for key, value in adata.layers.items()
        }

    if adata.obsm:
        serialized["obsm"] = {
            key: ensure_numpy_array(value) for key, value in adata.obsm.items()
        }

    if adata.varm:
        serialized["varm"] = {
            key: ensure_numpy_array(value) for key, value in adata.varm.items()
        }

    if adata.obsp:
        serialized["obsp"] = {
            key: ensure_numpy_array(value) for key, value in adata.obsp.items()
        }

    if adata.varp:
        serialized["varp"] = {
            key: ensure_numpy_array(value) for key, value in adata.varp.items()
        }

    if adata.uns:
        serialized["uns"] = adata.uns

    if adata.raw is not None:
        serialized["raw"] = serialize_anndata(adata.raw)

    logger.debug(
        "\nSerializing AnnData object from dictionary:\n\n"
        f"{pretty_log_dict(serialized)}\n\n"
    )

    return serialized


@beartype
def deserialize_anndata(
    data: Dict[str, Any],
    sparse_layers: bool = False,
) -> AnnData | AnnDataRaw:
    """
    Deserialize a dictionary to an AnnData object.

    Args:
        data: Dictionary representation of an AnnData object
        sparse_layers: If True, store layers as sparse matrices (csr_matrix)
                      instead of dense numpy arrays

    Returns:
        Reconstructed AnnData object
    """
    if not all(key in data for key in ["obs", "var", "shape"]):
        raise ValueError("Invalid data format: missing required keys")

    adata_dict = {
        "var": pd.DataFrame(data["var"]).set_index("index")
        if "index" in data["var"]
        else pd.DataFrame(index=range(data["shape"][1])),
        "obs": pd.DataFrame(data["obs"]).set_index("index")
        if "index" in data["obs"]
        else pd.DataFrame(index=range(data["shape"][0])),
    }

    if "X" in data:
        X = np.array(data["X"])
        if X.size == 0:
            X = X.reshape(data["shape"])
        adata_dict["X"] = X

    if "layers" in data:
        if sparse_layers:
            adata_dict["layers"] = {
                key: sparse.csr_matrix(np.array(value))
                for key, value in data["layers"].items()
            }
        else:
            adata_dict["layers"] = {
                key: np.array(value) for key, value in data["layers"].items()
            }

    if "obsm" in data:
        adata_dict["obsm"] = {
            key: np.array(value) for key, value in data["obsm"].items()
        }

    if "varm" in data:
        adata_dict["varm"] = {
            key: np.array(value) for key, value in data["varm"].items()
        }

    if "obsp" in data:
        adata_dict["obsp"] = {
            key: np.array(value) for key, value in data["obsp"].items()
        }

    if "varp" in data:
        adata_dict["varp"] = {
            key: np.array(value) for key, value in data["varp"].items()
        }

    if "uns" in data:
        adata_dict["uns"] = data["uns"]
        for key in [
            "clusters_coarse_colors",
            "clusters_colors",
            "day_colors",
            "velocity_graph",
            "velocity_graph_neg",
        ]:
            if key in adata_dict["uns"] and isinstance(
                adata_dict["uns"][key], list
            ):
                adata_dict["uns"][key] = np.array(adata_dict["uns"][key])

    adata = AnnData(**adata_dict)

    category_columns = ["clusters", "clusters_coarse", "leiden"]
    for col in category_columns:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype("category")

    if "highly_variable_genes" in adata.var.columns:
        adata.var["highly_variable_genes"] = adata.var[
            "highly_variable_genes"
        ].astype("category")

    if "raw" in data:
        raw_data = deserialize_anndata(data["raw"])
        if isinstance(raw_data, AnnData):
            adata.raw = raw_data
        else:
            adata.raw = AnnData(X=raw_data.X, var=raw_data.var)

    return adata if "obs" in data else AnnDataRaw(adata)


@beartype
def save_anndata_to_json(
    adata: AnnData,
    filename: str | Path,
    expected_hash: str | None = None,
) -> str:
    """
    Save an AnnData object to a JSON file.

    Args:
        adata: AnnData object to save
        filename: Name of the JSON file to save to
        expected_hash: Optional hash to validate against

    Returns:
        SHA-256 hash of the saved file
    """
    filename = Path(filename)
    adata_dict = serialize_anndata(adata)

    with filename.open("w") as f:
        json.dump(adata_dict, f, indent=4, cls=NumpyEncoder)

    file_hash = hash_file(filename)
    logger.info(f"\nSaved file: {filename}\nSHA-256 hash: {file_hash}\n")

    if expected_hash is not None:
        if file_hash == expected_hash:
            logger.info("Hash validation succeeded.")
        else:
            logger.warning(
                f"\nHash mismatch.\n"
                f"Expected: {expected_hash},\n"
                f"Actual: {file_hash}\n\n"
            )

    return file_hash


@beartype
def load_anndata_from_json(
    filename: str | Path,
    expected_hash: str | None = None,
    sparse_layers: bool = False,
) -> AnnData:
    """
    Load an AnnData object from a JSON file.

    Args:
        filename: Name of the JSON file to load from
        expected_hash: Optional hash to validate against
        sparse_layers: If True, store layers as sparse matrices

    Returns:
        Reconstructed AnnData object
    """
    filename = Path(filename)
    file_hash = hash_file(filename)
    logger.info(f"\nLoading file: {filename}\nSHA-256 hash: {file_hash}\n\n")

    if expected_hash is not None:
        if file_hash == expected_hash:
            logger.info("Hash validation succeeded.")
        else:
            logger.warning(
                f"\nHash mismatch.\n"
                f"Expected: {expected_hash}\n"
                f"Actual: {file_hash}\n\n"
            )

    with filename.open("r") as f:
        adata_dict = json.load(f)
    return deserialize_anndata(adata_dict, sparse_layers=sparse_layers)


@beartype
def create_sample_anndata(M: int, N: int) -> AnnData:
    """
    Create a sample AnnData object.

    Args:
        M: Number of observations
        N: Number of variables

    Returns:
        Sample AnnData object
    """
    adata_dict = {
        "X": np.random.random((M, N)),
        "obs": pd.DataFrame(
            {"batch": np.random.choice(["a", "b"], M)},
            index=[f"cell{i:03d}" for i in range(M)],
        ),
        "var": pd.DataFrame(index=[f"gene{i:03d}" for i in range(N)]),
        "obsm": {
            "a": np.random.random((M, 100)),
        },
        "layers": {
            "a": np.random.random((M, N)),
        },
        "uns": {"sample_key": "sample_value"},
    }
    return AnnData(**adata_dict)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, sparse.spmatrix):
            return obj.todense().tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_python(obj):
    if isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif sparse.issparse(obj):
        return obj.todense().tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj
