import os
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote

import anndata
import anndata._core.anndata
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import requests
import scanpy as sc
import scvelo as scv
import validators
from beartype import beartype
from scipy.sparse import issparse

from pyrovelocity.cytotrace import cytotrace_sparse
from pyrovelocity.logging import configure_logging
from pyrovelocity.utils import (
    ensure_numpy_array,
    generate_sample_data,
    print_anndata,
    print_attributes,
)

logger = configure_logging(__name__)


@beartype
def download_dataset(
    data_set_name: str,
    download_file_name: str,
    download_path_root: str = "data/external",
    data_external_path: str = "data/external",
    source: Optional[str] = None,
    data_url: Optional[str] = None,
) -> Path:
    """
    Downloads a dataset based on the specified parameters and returns the path
    to the downloaded data.

    Args:
        data_set_name (str): Name of the dataset to download.
        download_file_name (str): Name of the dataset to download.
        download_path_root (Path): Path for downloading the dataset. Default is 'data/external'.
        data_external_path (Path): Path where the downloaded data will be stored. Default is 'data/external'.
        source (str): The source type of the dataset. Default is 'simulate'.
        data_url (str): URL from where the dataset can be downloaded.

    Returns:
        Path: The path to the downloaded dataset file.

    Examples:
        >>> tmp = getfixture('tmp_path')
        >>> simulate_medium_dataset = download_dataset(
        ...   'simulated_medium',
        ...   'simulated_medium',
        ...   str(tmp) + 'data/external',
        ...   str(tmp) + 'data/external',
        ...   'simulate',
        ... ) # xdoctest: +SKIP
        >>> pancreas_dataset = download_dataset(
        ...   'pancreas',
        ...   'endocrinogenesis_day15',
        ...   str(tmp) + 'data/Pancreas',
        ...   str(tmp) + 'data/external',
        ...   'scvelo',
        ...   'https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad',
        ... ) # xdoctest: +SKIP
        >>> pancreas_dataset = download_dataset(
        ...   'pancreas',
        ...   'pancreas',
        ...   str(tmp) + 'data/external',
        ...   str(tmp) + 'data/external',
        ...   'scvelo',
        ... ) # xdoctest: +SKIP
    """
    download_path = Path(download_path_root) / f"{download_file_name}.h5ad"
    data_path = Path(data_external_path) / f"{data_set_name}.h5ad"
    data_external_path = Path(data_external_path)

    logger.info(
        f"\n\nVerifying existence of path for downloaded data: {data_external_path}\n"
    )
    data_external_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"\n\nVerifying {data_set_name} data:\n"
        f"  data will be temporarily downloaded to {download_path}\n"
        f"  and stored in {data_path}\n"
    )

    if data_path.is_file() and os.access(str(data_path), os.R_OK):
        logger.info(f"{data_path} exists")
        return data_path
    else:
        logger.info(f"Attempting to download {data_set_name} data...")
        if data_url is not None:
            logger.info(f"Validating URL {data_url}...")
            is_valid_url, valid_url_message = validate_url_and_file(data_url)
            if not is_valid_url:
                raise ValueError(
                    f"Invalid URL: {data_url}\n{valid_url_message}\n"
                )
            else:
                logger.info(valid_url_message)
            try:
                adata = sc.read(str(data_path), backup_url=data_url)
            except Exception as e:
                logger.warn(f"Failed to download from URL {data_url}: {e}")
            else:
                adata.write(str(data_path))
                print_attributes(adata)
                print_anndata(adata)
                return data_path

        if source == "scvelo":
            logger.info(f"Downloading {data_set_name} data from scvelo...")
            dl_method = getattr(scv.datasets, data_set_name)
            adata = dl_method()
            adata.write(str(data_path))
        elif source == "simulate":
            logger.info(f"Generating {data_set_name} data from simulation...")
            adata = generate_sample_data(
                n_obs=3000,
                n_vars=1000,
                noise_model="gillespie",
                random_seed=99,
            )
            adata.write(str(data_path))

        print_attributes(adata)
        print_anndata(adata)

        if download_path != data_path:
            download_path.replace(data_path)
            try:
                download_path.rmdir()
            except OSError as e:
                logger.warn(f"{download_path} : {e.strerror}")

        if data_path.is_file() and os.access(str(data_path), os.R_OK):
            logger.info(f"successfully downloaded {data_path}")
        else:
            logger.warn(f"cannot find and read {data_path}")

        return data_path


@beartype
def validate_url_and_file(url: str) -> Tuple[bool, str]:
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


def copy_raw_counts(
    adata: anndata._core.anndata.AnnData,
) -> anndata._core.anndata.AnnData:
    """
    Copy unspliced and spliced raw counts to adata.layers and adata.obs.

    Args:
        adata (anndata._core.anndata.AnnData): AnnData object

    Returns:
        anndata._core.anndata.AnnData: AnnData object with raw counts.

    Examples:
        >>> from pyrovelocity.utils import generate_sample_data
        >>> adata = generate_sample_data()
        >>> copy_raw_counts(adata)
    """
    adata.layers["raw_unspliced"] = adata.layers["unspliced"]
    print("'raw_unspliced', raw unspliced counts (adata.layers)")
    adata.layers["raw_spliced"] = adata.layers["spliced"]
    print("'raw_spliced', raw spliced counts (adata.layers)")
    adata.obs["u_lib_size_raw"] = (
        adata.layers["raw_unspliced"].toarray().sum(-1)
        if issparse(adata.layers["raw_unspliced"])
        else adata.layers["raw_unspliced"].sum(-1)
    )
    print("'u_lib_size_raw', unspliced library size (adata.obs)")
    adata.obs["s_lib_size_raw"] = (
        adata.layers["raw_spliced"].toarray().sum(-1)
        if issparse(adata.layers["raw_spliced"])
        else adata.layers["raw_spliced"].sum(-1)
    )
    print("'s_lib_size_raw', spliced library size (adata.obs)")
    return adata


def assign_colors(
    max_spliced: int, max_unspliced: int, minlim_s: int, minlim_u: int
) -> List[str]:
    return [
        "black"
        if (spliced >= minlim_s) & (unspliced >= minlim_u)
        else "lightgrey"
        for spliced, unspliced in zip(max_spliced, max_unspliced)
    ]


def get_thresh_histogram_title_from_path(path):
    title = os.path.basename(path)
    title = os.path.splitext(title)[0]
    title = title.replace("_thresh_histogram", "")
    return title.replace("_", " ")


def plot_high_us_genes(
    adata: anndata.AnnData,
    thresh_histogram_path: str,
    minlim_u: int = 3,
    minlim_s: int = 3,
    unspliced_layer: str = "unspliced",
    spliced_layer: str = "spliced",
) -> Optional[matplotlib.figure.Figure]:
    if (
        adata is None
        or unspliced_layer not in adata.layers
        or spliced_layer not in adata.layers
    ):
        raise ValueError(
            "Invalid data set. Please ensure that adata is an AnnData object"
            "and that the layers 'unspliced' and 'spliced' are present."
        )

    max_unspliced = np.array(
        np.max(ensure_numpy_array(adata.layers[unspliced_layer]), axis=0)
    ).flatten()
    max_spliced = np.array(
        np.max(ensure_numpy_array(adata.layers[spliced_layer]), axis=0)
    ).flatten()

    ### create figure
    x = max_spliced
    y = max_unspliced

    title = get_thresh_histogram_title_from_path(thresh_histogram_path)

    colors = assign_colors(max_spliced, max_unspliced, minlim_s, minlim_u)

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    fig = plt.figure(figsize=(4, 4))
    fig.suptitle(title, y=1.00, fontsize=12)

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ### the scatter plot:
    ax.scatter(max_spliced, max_unspliced, s=1, c=colors)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("max. spliced counts")
    ax.set_ylabel("max. unspliced counts")
    if minlim_s > 0:
        ax.axhline(y=minlim_s - 1, color="r", linestyle="--")
    if minlim_u > 0:
        ax.axvline(x=minlim_u - 1, color="r", linestyle="--")

    ### the histograms:
    bins = 50
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(0, np.log10(bins[-1]), len(bins))
    ax_histx.hist(x, bins=logbins)
    ax_histy.hist(y, bins=logbins, orientation="horizontal")

    for ext in ["", ".png"]:
        fig.savefig(
            f"{thresh_histogram_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    return fig


def get_high_us_genes(
    adata,
    minlim_u=0,
    minlim_s=0,
    unspliced_layer="unspliced",
    spliced_layer="spliced",
):
    """
    Function to select genes that have spliced and unspliced counts above a
    certain threshold. Genes of which the maximum u and s count is above a set
    threshold are selected. Threshold varies per dataset and influences the
    numbers of genes that are selected.

    Parameters
    ----------
    adata
        Annotated data matrix
    minlim_u: `int` (default: 3)
        Threshold above which the maximum unspliced counts of a gene should fall to be included in the
        list of high US genes.
    minlim_s: `int` (default: 3)
        Threshold above which the maximum spliced counts of a gene should fall to be included in the
        list of high US genes.
    unspliced_layer: `str` (default: 'unspliced')
        Name of layer that contains the unspliced counts.
    spliced_layer: `str` (default: 'spliced')
        Name of layer that contains the spliced counts.
    """
    print("adata.shape before filtering:", adata.shape)
    from scipy import sparse

    # test if layers are not sparse but dense
    for layer in [unspliced_layer, spliced_layer]:
        if sparse.issparse(adata.layers[layer]):
            adata.layers[layer] = adata.layers[layer].todense()

    # get high US genes
    u_genes = np.max(adata.layers[unspliced_layer], axis=0) >= minlim_u
    s_genes = np.max(adata.layers[spliced_layer], axis=0) >= minlim_s
    us_genes = adata.var_names[np.array(u_genes & s_genes).flatten()].values
    adata = adata[:, us_genes]
    for layer in [unspliced_layer, spliced_layer]:
        adata.layers[layer] = sparse.csr_matrix(adata.layers[layer])
    print("adata.shape after filtering:", adata.shape)
    return adata


def load_data(
    data: str = "pancreas",
    top_n: int = 2000,
    min_shared_counts: int = 30,
    eps: float = 1e-6,
    force: bool = False,
    processed_path: str = None,
    process_cytotrace: bool = False,
    use_sub: bool = False,
    count_thres: int = 0,
    thresh_histogram_path: str = None,
) -> anndata._core.anndata.AnnData:
    """
    Preprocess data from scvelo.

    Args:
        data (str, optional): data set name. Defaults to scvelo's "pancreas" data set.
        top_n (int, optional): number of genes to retain. Defaults to 2000.
        min_shared_counts (int, optional): minimum shared counts. Defaults to 30.
        eps (float, optional): tolerance. Defaults to 1e-6.
        force (bool, optional): force reprocessing. Defaults to False.
        processed_path (str, optional): path to read/write processed AnnData. Defaults to None.

    Returns:
        anndata._core.anndata.AnnData: processed AnnData object
    """
    if processed_path is None:
        processed_path = f"{data}_processed.h5ad"
    print(
        processed_path,
        os.path.isfile(processed_path),
        os.access(processed_path, os.R_OK),
        (not force),
    )
    if (
        os.path.isfile(processed_path)
        and os.access(processed_path, os.R_OK)
        and (not force)
    ):
        adata = sc.read(processed_path)
    else:
        print("Dataset name:", data)
        if data == "pancreas":
            adata = scv.datasets.pancreas()
        elif data == "bonemarrow":
            adata = scv.datasets.bonemarrow()
        elif data == "forebrain":
            adata = scv.datasets.forebrain()
        elif data == "dentategyrus_lamanno":
            adata = scv.datasets.dentategyrus_lamanno()
        elif data == "dentategyrus":
            adata = scv.datasets.dentategyrus()
        elif "larry" in data:
            data = data.split("/")[-1].split(".")[0]
            print("Larry dataset name:", data)
            if data == "larry":
                adata = load_larry()
            elif data == "larry_tips":
                adata = load_larry()
                adata = adata[adata.obs["time_info"] == 6.0]
                adata = adata[adata.obs["state_info"] != "Undifferentiated"]
            elif data in ["larry_mono", "larry_neu"]:
                adata = load_unipotent_larry(data.split("_")[1])
                adata = adata[adata.obs.state_info != "Centroid", :]
            elif data == "larry_multilineage":
                adata_mono = load_unipotent_larry("mono")
                adata_mono_C = adata_mono[
                    adata_mono.obs.state_info != "Centroid", :
                ].copy()
                adata_neu = load_unipotent_larry("neu")
                adata_neu_C = adata_neu[
                    adata_neu.obs.state_info != "Centroid", :
                ].copy()
                adata_multilineage = adata_mono.concatenate(adata_neu)
                adata = adata_mono_C.concatenate(adata_neu_C)
            elif data in ["larry_cospar", "larry_cytotrace", "larry_dynamical"]:
                adata = sc.read(f"data/external/{data}.h5ad")
                print_anndata(adata)
                copy_raw_counts(adata)
                print_anndata(adata)
                # if count_thres:
                plot_high_us_genes(
                    adata=adata,
                    thresh_histogram_path=thresh_histogram_path,
                    minlim_u=count_thres,
                    minlim_s=count_thres,
                    unspliced_layer="raw_unspliced",
                    spliced_layer="raw_spliced",
                )
                adata.write(processed_path)
                return adata
        elif "pbmc68k" in data:
            adata = load_pbmc68k(
                data,
                count_thres=count_thres,
                thresh_histogram_path=thresh_histogram_path,
            )
        else:  # pbmc10k
            adata = sc.read(data)

        print_anndata(adata)
        if "raw_unspliced" not in adata.layers:
            copy_raw_counts(adata)
            print_anndata(adata)

        if process_cytotrace:
            print("Processing data with cytotrace ...")
            cytotrace_sparse(adata, layer="spliced")

        if not "pbmc68k" in data:
            scv.pp.filter_and_normalize(
                adata, min_shared_counts=min_shared_counts, n_top_genes=top_n
            )
            # if count_thres:
            plot_high_us_genes(
                adata=adata,
                thresh_histogram_path=thresh_histogram_path,
                minlim_u=count_thres,
                minlim_s=count_thres,
                unspliced_layer="raw_unspliced",
                spliced_layer="raw_spliced",
            )
            adata = get_high_us_genes(
                adata,
                minlim_u=count_thres,
                minlim_s=count_thres,
                unspliced_layer="raw_unspliced",
                spliced_layer="raw_spliced",
            )
            scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            scv.tl.recover_dynamics(adata, n_jobs=-1, use_raw=False)
            scv.tl.velocity(adata, mode="dynamical", use_raw=False)
        if ("X_umap" not in adata.obsm.keys()) or (data == "larry_tips"):
            scv.tl.umap(adata)
        if "leiden" not in adata.obs.keys():
            sc.tl.leiden(adata)
        if use_sub:
            top_genes = (
                adata.var["fit_likelihood"].sort_values(ascending=False).index
            )
            print(top_genes[:10])
            adata = adata[:, top_genes[:3]].copy()
        scv.tl.velocity_graph(adata, n_jobs=-1)

        if data in ["larry", "larry_mono", "larry_neu", "larry_multilineage"]:
            scv.tl.velocity_embedding(adata, basis="emb")
        else:
            scv.tl.velocity_embedding(adata)

        scv.tl.latent_time(adata)

        print_anndata(adata)
        adata.write(processed_path)

    return adata


def load_pbmc68k(
    data: str = "pbmc68k",  # pbmc68k or pbmc10k
    count_thres: int = 0,
    thresh_histogram_path: str = None,
) -> anndata._core.anndata.AnnData:
    if data == "pbmc68k":
        adata = scv.datasets.pbmc68k()
    elif os.path.isfile(data) and os.access(data, os.R_OK):
        adata = sc.read(data)

    print_anndata(adata)
    if "raw_unspliced" not in adata.layers:
        copy_raw_counts(adata)
        print_anndata(adata)
    print("Removing duplicate cells and tSNE x-parity in pbmc68k data...")
    scv.pp.remove_duplicate_cells(adata)
    adata.obsm["X_tsne"][:, 0] *= -1
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
    # if count_thres:
    plot_high_us_genes(
        adata,
        thresh_histogram_path,
        minlim_u=count_thres,
        minlim_s=count_thres,
        unspliced_layer="raw_unspliced",
        spliced_layer="raw_spliced",
    )
    adata = get_high_us_genes(
        adata,
        minlim_u=count_thres,
        minlim_s=count_thres,
        unspliced_layer="raw_unspliced",
        spliced_layer="raw_spliced",
    )
    scv.pp.moments(adata)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.recover_dynamics(adata, n_jobs=-1)

    return adata


def load_larry(
    file_path: str = "data/external/larry.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro Hematopoiesis LARRY datasets.

    Data from `CALEB WEINREB et al. (2020) <DOI: 10.1126/science.aaw3381>'
    https://figshare.com/ndownloader/articles/20780344/versions/1

    Returns
    -------
    Returns `adata` object
    """
    url = "https://figshare.com/ndownloader/files/37028569"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


def load_unipotent_larry(
    celltype: str = "mono"
) -> anndata._core.anndata.AnnData:
    """In vitro Hemotopoiesis Larry datasets
    Subset of Data from `CALEB WEINREB et al. (2020) <DOI: 10.1126/science.aaw3381>'
    unipotent monocytes: https://figshare.com/ndownloader/files/37028572
    unipotent neutrophils: https://figshare.com/ndownloader/files/37028575

    Returns
    -------
    Returns `adata` object
    """
    file_path = f"data/external/larry_{celltype}.h5ad"
    if celltype == "mono":
        url = "https://figshare.com/ndownloader/files/37028572"
    else:  # neutrophil
        url = "https://figshare.com/ndownloader/files/37028575"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata
