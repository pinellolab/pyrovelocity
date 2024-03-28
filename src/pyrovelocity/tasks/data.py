import inspect
import os
from pathlib import Path
from typing import Optional
from typing import Tuple
from urllib.parse import unquote

import anndata
import numpy as np
import requests
import scanpy as sc
import validators
from anndata._core.anndata import AnnData
from beartype import beartype

import pyrovelocity.io.datasets
from pyrovelocity.logging import configure_logging
from pyrovelocity.utils import generate_sample_data
from pyrovelocity.utils import print_anndata


__all__ = ["download_dataset", "load_anndata_from_path", "subset_anndata"]

logger = configure_logging(__name__)


@beartype
def download_dataset(
    data_set_name: str,
    data_external_path: str = "data/external",
    source: str = "pyrovelocity",
    data_url: Optional[str] = None,
    n_obs: Optional[int] = None,
    n_vars: Optional[int] = None,
) -> Path:
    """
    Downloads a dataset based on the specified parameters and returns the path
    to the downloaded data.

    Args:
        data_set_name (str): Name of the dataset to download.
        download_file_name (str): Name of the dataset to download.
        download_path_root (Path): Path for downloading the dataset. Default is 'data/external'.
        data_external_path (Path): Path where the downloaded data will be stored. Default is 'data/external'.
        source (str): The source type of the dataset. Default is 'pyrovelocity'.
        data_url (str): URL from where the dataset can be downloaded. Takes precedence over source.
        n_obs (int): Number of observations to sample from the dataset. Defaults to None.
        n_vars (int): Number of variables to sample from the dataset. Defaults to None.

    Returns:
        Path: The path to the downloaded dataset file.

    Examples:
        >>> tmp = getfixture('tmp_path')
        >>> simulated_dataset = download_dataset(
        ...   'simulated',
        ...   str(tmp) + '/data/external',
        ...   'simulate',
        ...   n_obs=100,
        ...   n_vars=300,
        ... ) # xdoctest: +SKIP
        >>> simulated_dataset = download_dataset(
        ...   'simulated_path',
        ...   tmp / Path('data/external'),
        ...   'simulate',
        ...   n_obs=100,
        ...   n_vars=300,
        ... ) # xdoctest: +SKIP
        >>> pancreas_dataset = download_dataset(
        ...   data_set_name='pancreas_direct',
        ...   data_external_path=str(tmp) + '/data/external',
        ...   data_url='https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad',
        ... ) # xdoctest: +SKIP
        >>> pancreas_dataset = download_dataset(
        ...   'pancreas',
        ...   str(tmp) + '/data/external',
        ...   'pyrovelocity',
        ... ) # xdoctest: +SKIP
        >>> no_dataset = download_dataset(
        ...   'no_dataset',
        ...   str(tmp) + '/data/external',
        ...   'pyrovelocity',
        ... ) # xdoctest: +SKIP, +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        AttributeError
    """
    adata = None
    data_path = Path(data_external_path) / f"{data_set_name}.h5ad"
    data_external_path = Path(data_external_path)

    logger.info(
        f"\n\nVerifying existence of path for downloaded data: {data_external_path}\n"
    )
    data_external_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"\n\nVerifying {data_set_name} data:\n"
        f"  data will be stored in {data_path}\n"
    )

    if data_path.is_file() and os.access(str(data_path), os.R_OK):
        logger.info(f"{data_path} exists")
        return data_path
    else:
        logger.info(f"Attempting to download {data_set_name} data...")
        if data_url is not None:
            logger.info(f"Validating URL {data_url}...")
            is_valid_url, valid_url_message = _validate_url_and_file(data_url)
            if not is_valid_url:
                raise ValueError(
                    f"Invalid URL: {data_url}\n{valid_url_message}\n"
                )
            else:
                logger.info(valid_url_message)
            try:
                adata = sc.read(str(data_path), backup_url=data_url, cache=True)
            except Exception as e:
                logger.error(f"Failed to download from URL {data_url}: {e}")
        elif source == "pyrovelocity":
            logger.info(
                f"Downloading {data_set_name} data with pyrovelocity..."
            )
            try:
                download_method = getattr(
                    pyrovelocity.io.datasets, data_set_name
                )
            except AttributeError:
                available_datasets = [
                    name
                    for name, func in inspect.getmembers(
                        pyrovelocity.io.datasets, inspect.isfunction
                    )
                    if func.__module__ == pyrovelocity.io.datasets.__name__
                ]
                error_message = (
                    f"Dataset '{data_set_name}' not found in pyrovelocity.datasets. "
                    f"Available datasets are: {available_datasets}"
                    f"You can specify a URL that resolves to a .h5ad file with"
                    f"the `data_url` argument instead of a source."
                )
                logger.error(error_message)
                raise
            else:
                adata = download_method(file_path=data_path)
                if n_obs is not None and n_vars is not None:
                    adata, _ = subset_anndata(
                        adata=adata, n_obs=n_obs, n_vars=n_vars
                    )
                    adata.write(data_path)
                elif n_obs is not None:
                    adata, _ = subset_anndata(adata=adata, n_obs=n_obs)
                    adata.write(data_path)
                elif n_vars is not None:
                    logger.warning("n_vars is ignored if n_obs is not provided")

        elif source == "simulate":
            if n_obs is None or n_vars is None:
                raise ValueError(
                    "n_obs and n_vars must be provided if source is 'simulate'"
                )
            logger.info(f"Generating {data_set_name} data from simulation...")
            adata = generate_sample_data(
                n_obs=n_obs,
                n_vars=n_vars,
                noise_model="gillespie",
                random_seed=99,
            )
            adata.write(str(data_path))
        else:
            raise ValueError(
                f"Invalid source: {source}\n"
                f"or URL: {data_url}\n"
                f"Please specify a valid source or URL that resolves to a .h5ad file."
            )

        if adata:
            print_anndata(adata)
            if data_path.is_file() and os.access(str(data_path), os.R_OK):
                logger.info(f"Successfully downloaded {data_path}")
            else:
                logger.error(f"Cannot find and read {data_path}")
        else:
            logger.error(
                f"No data available for {data_set_name} due to errors during download or processing."
            )

        return data_path


@beartype
def _validate_url_and_file(url: str) -> Tuple[bool, str]:
    """
    Validates a given URL format and checks if it leads to a .h5ad file larger
    than 1MB.

    This function first checks if the URL is valid using the
    validators.url method. Then, it makes a HEAD request to the URL to fetch
    headers without downloading the entire file. It checks the Content-Type and
    Content-Disposition headers for the .h5ad file extension and the Content-
    Length header to determine the file size.

    Args:
        url (str): The URL to validate and check the file type and size.

    Returns:
        tuple: A tuple containing a boolean and a validation message.
               The boolean is True if the URL is valid, leads to a .h5ad file, and the file is larger than 1MB.
               The message provides additional information about the validation.

    Raises:
        requests.RequestException: If an error occurs during the HEAD request.

    Examples:
        >>> is_valid, message = validate_url_and_file("https://storage.googleapis.com/pyrovelocity/data/pbmc5k.h5ad")
        >>> logger.info(f"valid: {is_valid}\n{message}")

        >>> is_valid, message = validate_url_and_file("http?/invalid.url/file.txt")
        >>> logger.info(f"valid: {is_valid}\n{message}")

        >>> is_valid, message = validate_url_and_file("https://invalid.url/file.txt")
        >>> logger.info(f"valid: {is_valid}\n{message}")
    """
    if not validators.url(url):
        return False, "Invalid URL format"

    try:
        response = requests.head(url, allow_redirects=True)
        response.raise_for_status()

        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            filename = content_disposition.split("filename=")[-1]
            filename = unquote(filename).strip('"')
        else:
            filename = url.split("/")[-1]

        if not filename.endswith(".h5ad"):
            return False, "The file does not have an .h5ad extension"

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) <= 1_000_000:
            return False, "The file size is less than or equal to 1MB"
        else:
            content_length_mb = int(content_length) / 1_000_000

        return (
            True,
            f"URL validated and file is an .h5ad file named {filename} with size {content_length_mb:.1f} MB",
        )
    except requests.RequestException as e:
        return False, f"Error occurred: {e}"


@beartype
def subset_anndata(
    file_path: Optional[str | Path] = None,
    adata: Optional[anndata._core.anndata.AnnData] = None,
    n_obs: int = 100,
    n_vars: Optional[int] = None,
    save_subset: bool = False,
    output_path: Optional[str | Path] = None,
) -> Tuple[anndata._core.anndata.AnnData, str | Path | None]:
    """
    Randomly sample observations from a dataset given by file path or AnnData object.

    Args:
        file_path (str): Path to a .h5ad file containing a dataset. Takes precedence over adata.
        adata (AnnData): AnnData object. If None, file_path must be provided.
        n_obs (int): Number of observations to sample. Defaults to 100.
        save_subset (bool): If True, save the subset to a file. Defaults to False.
        output_path (str): Path to save the subset. Defaults to None.

    Raises:
        ValueError: If neither file_path nor adata is provided.

    Returns:
        AnnData: Subset of the dataset.

    Examples:
        >>> data_path = download_dataset(data_set_name="pancreas") # xdoctest: +SKIP
        >>> adata = subset(file_path=data_path, n_obs=100, save_subset=True) # xdoctest: +SKIP
        >>> print_anndata(adata) # xdoctest: +SKIP
        >>> print_attributes(adata) # xdoctest: +SKIP
    """
    if file_path is not None:
        file_path = Path(file_path)
        adata = sc.read(file_path, cache=True)
    if adata is None:
        raise ValueError("Either file_path or adata must be provided")

    if n_obs > adata.n_obs:
        logger.warning(
            f"n_obs ({n_obs}) is greater than the number of observations in the dataset ({adata.n_obs})"
        )
        n_obs = adata.n_obs
    logger.info("constructing data subset")
    print_anndata(adata)

    if n_vars is not None:
        if n_vars > adata.n_vars:
            logger.warning(
                f"n_vars ({n_vars}) is greater than the number of variables in the dataset ({adata.n_vars})"
            )
            n_vars = adata.n_vars
        selected_vars_indices = np.random.choice(adata.n_vars, n_vars)
        logger.info(f"selected {n_vars} vars from {adata.n_vars}")
        adata = adata[:, selected_vars_indices]

    selected_obs_indices = np.random.choice(adata.n_obs, n_obs)
    logger.info(f"selected {n_obs} obs from {adata.n_obs}")
    adata = adata[selected_obs_indices]
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if save_subset:
        if output_path is None and file_path is not None:
            output_path = file_path.parent / Path(
                file_path.stem + f"_{n_obs}obs" + file_path.suffix
            )
        if output_path is None:
            raise ValueError(
                "output_path must be provided if save_subset is True and file_path is None"
            )
        adata.write(output_path)
        logger.info(f"saved {n_obs} obs subset: {output_path}")

    print_anndata(adata)
    return adata.copy(), output_path


@beartype
def load_anndata_from_path(adata_path: str | Path) -> AnnData:
    adata_path = Path(adata_path)
    if adata_path.suffix not in {".h5ad", ".loom"}:
        raise ValueError(
            f"The input file {adata_path}\n"
            "must be either a .h5ad or .loom file."
        )
    if os.path.isfile(adata_path) and os.access(adata_path, os.R_OK):
        logger.info(f"Reading input file: {adata_path}")
        adata = sc.read(filename=adata_path, cache=True)
        return adata
    else:
        raise ValueError(f"Cannot read input file: {adata_path}")
