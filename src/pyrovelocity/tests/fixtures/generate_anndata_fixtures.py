"""
Execute this script to generate test fixture data:

python src/pyrovelocity/tests/fixtures/generate_anndata_fixtures.py

This script should not be executed by pytest.
It is used to generate test fixture data.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
from anndata import AnnData
from beartype import beartype
from beartype.typing import List

from pyrovelocity.io.datasets import pancreas
from pyrovelocity.io.serialization import (
    load_anndata_from_json,
    save_anndata_to_json,
)
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.utils import (
    anndata_string,
    configure_logging,
    load_anndata_from_path,
    print_anndata,
    print_string_diff,
)
from pyrovelocity.workflows.main_configuration import (
    pancreas_summary_configuration,
)

__all__ = ["generate_pancreas_fixture_data"]

logger = configure_logging(__name__)


@beartype
def generate_pancreas_fixture_data(
    output_path: str
    | Path = "src/pyrovelocity/tests/data/preprocessed_pancreas.json",
    n_obs: int = 50,
    n_vars: int = 7,
) -> Path:
    """
    Generate a test fixture for the pancreas dataset.

    Note that the selected_genes are not enforced in the preprocess_dataset
    function. This is to prefer higher quality genes over the selected_genes
    in generating the test fixture data.

    Args:
        output_path: Path to save the JSON fixture.
        n_obs: Number of observations to keep.
        n_vars: Number of variables to keep.
    """
    output_path = Path(output_path)

    adata: AnnData = pancreas()

    adata, _, _ = preprocess_dataset(
        data_set_name="pancreas",
        adata=adata,
        use_obs_subset=True,
        n_obs_subset=n_obs,
        use_vars_subset=True,
        n_vars_subset=n_vars,
        process_cytotrace=True,
    )

    selected_genes: List[str] = pancreas_summary_configuration.selected_genes
    selected_genes_in_adata: List[str] = [
        gene for gene in selected_genes if gene in adata.var_names
    ]
    if len(selected_genes_in_adata) < len(selected_genes):
        logger.warning(
            f"\nSome selected genes are not in the downsampled data:\n"
            f"selected_genes: {selected_genes}\n"
            f"selected_genes_in_adata: {selected_genes_in_adata}\n"
        )

    genes_to_keep: List[str] = list(
        set(adata.var_names[:n_vars].tolist() + selected_genes_in_adata)
    )
    adata = adata[:, genes_to_keep]
    preprocessed_anndata_string = anndata_string(adata)

    print_anndata(adata)
    save_anndata_to_json(adata, output_path)

    logger.info(f"Test fixture saved to {output_path}")

    try:
        logger.info("Attempting to load the serialized AnnData object...")
        loaded_adata = load_anndata_from_json(output_path)
        loaded_anndata_string = anndata_string(loaded_adata)
        logger.info("Successfully loaded the serialized AnnData object.")
        print_string_diff(
            text1=preprocessed_anndata_string,
            text2=loaded_anndata_string,
            diff_title="Preprocessed vs Loaded AnnData",
        )
        print_anndata(loaded_adata)
    except Exception as e:
        logger.error(f"Error loading serialized AnnData object: {str(e)}")

    return output_path


@beartype
def generate_postprocessed_pancreas_fixture_data(
    input_path: str
    | Path = "src/pyrovelocity/tests/data/preprocessed_pancreas_50_7.json",
    trained_output_path: str
    | Path = "src/pyrovelocity/tests/data/trained_pancreas_50_7.json",
    postprocessed_output_path: str
    | Path = "src/pyrovelocity/tests/data/postprocessed_pancreas_50_7.json",
    max_epochs: int = 10,
    retain_temp_files: bool = False,
    retain_dir: str
    | Path = "src/pyrovelocity/tests/data/train_postprocess_artifacts",
) -> tuple[Path, Path]:
    """
    Generate trained and postprocessed test fixtures for the pancreas dataset.

    Args:
        input_path: Path to load the preprocessed JSON fixture.
        trained_output_path: Path to save the trained JSON fixture.
        postprocessed_output_path: Path to save the postprocessed JSON fixture.
        max_epochs: Number of epochs to train the model.
        retain_temp_files: If True, copy temporary files to retain_dir.
        retain_dir: Directory to copy temporary files to if retain_temp_files is True.

    Returns:
        Tuple of paths to the saved trained and postprocessed JSON fixtures.
    """
    input_path = Path(input_path)
    trained_output_path = Path(trained_output_path)
    postprocessed_output_path = Path(postprocessed_output_path)

    adata = load_anndata_from_json(input_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        models_path = Path(tmp_dir) / "models"
        logger.info(f"Using temporary directory: {models_path}")

        result = train_dataset(
            adata=adata,
            data_set_name="pancreas",
            model_identifier="model2",
            models_path=models_path,
            max_epochs=max_epochs,
            force=True,
        )

        (
            data_model,
            data_model_path,
            trained_data_path,
            model_path,
            posterior_samples_path,
            metrics_path,
            run_info_path,
            loss_plot_path,
            loss_csv_path,
        ) = result

        trained_adata = load_anndata_from_path(trained_data_path)

        (
            pyrovelocity_data_path,
            postprocessed_data_path,
        ) = postprocess_dataset(
            data_model=data_model,
            data_model_path=data_model_path,
            trained_data_path=trained_data_path,
            model_path=model_path,
            posterior_samples_path=posterior_samples_path,
            metrics_path=metrics_path,
            vector_field_basis="umap",
            number_posterior_samples=4,
        )

        postprocessed_adata = load_anndata_from_path(postprocessed_data_path)

        if retain_temp_files:
            retain_dir = Path(retain_dir)
            retain_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying temporary files to {retain_dir}")
            shutil.copytree(tmp_dir, retain_dir, dirs_exist_ok=True)

    preprocessed_anndata_string = anndata_string(adata)
    trained_anndata_string = anndata_string(trained_adata)
    postprocessed_anndata_string = anndata_string(postprocessed_adata)

    print_string_diff(
        text1=preprocessed_anndata_string,
        text2=trained_anndata_string,
        diff_title="Preprocessed vs Trained AnnData",
    )
    print_string_diff(
        text1=trained_anndata_string,
        text2=postprocessed_anndata_string,
        diff_title="Trained vs Postprocessed AnnData",
    )
    print_anndata(postprocessed_adata)

    save_anndata_to_json(trained_adata, trained_output_path)
    logger.info(f"Trained test fixture saved to {trained_output_path}")
    save_anndata_to_json(postprocessed_adata, postprocessed_output_path)
    logger.info(
        f"Postprocessed test fixture saved to {postprocessed_output_path}"
    )

    return trained_output_path, postprocessed_output_path


@beartype
def generate_larry_fixture_data(
    output_path: str
    | Path = "src/pyrovelocity/tests/data/larry_multilineage_50_6.json",
    n_obs: int = 50,
    genes: List[str] = ["Itgb2", "S100a9", "Fcer1g", "Lilrb4", "Vim", "Serbp1"],
) -> Path:
    """
    Generate a test fixture for the Larry multilineage dataset with specific genes.

    Args:
        output_path: Path to save the JSON fixture.
        n_obs: Number of observations to keep.
        genes: List of gene names to include. Defaults to a set of marker genes.

    Returns:
        Path to the saved fixture.
    """
    output_path = Path(output_path)

    from pyrovelocity.io.datasets import larry_multilineage

    adata: AnnData = larry_multilineage()

    available_genes = [gene for gene in genes if gene in adata.var_names]
    if len(available_genes) < len(genes):
        missing_genes = set(genes) - set(available_genes)
        logger.warning(
            f"Some requested genes are not in the dataset: {missing_genes}"
        )

    if not available_genes:
        logger.error("None of the requested genes are in the dataset!")
        var_indices = np.random.choice(
            adata.n_vars, size=min(6, adata.n_vars), replace=False
        )
        adata = adata[:, var_indices]
    else:
        adata = adata[:, available_genes]

    np.random.seed(42)
    obs_indices = np.random.choice(
        adata.n_obs, size=min(n_obs, adata.n_obs), replace=False
    )
    adata = adata[obs_indices, :]

    original_anndata_string = anndata_string(adata)
    print_anndata(adata)

    save_anndata_to_json(adata, output_path)
    logger.info(f"Larry multilineage test fixture saved to {output_path}")

    try:
        logger.info("Attempting to load the serialized Larry AnnData object...")
        loaded_adata = load_anndata_from_json(output_path)
        loaded_anndata_string = anndata_string(loaded_adata)
        logger.info("Successfully loaded the serialized Larry AnnData object.")
        print_string_diff(
            text1=original_anndata_string,
            text2=loaded_anndata_string,
            diff_title="Original vs Loaded Larry AnnData",
        )
        print_anndata(loaded_adata)
    except Exception as e:
        logger.error(f"Error loading serialized Larry AnnData object: {str(e)}")

    return output_path


if __name__ == "__main__":
    generate_pancreas_fixture_data()
    generate_postprocessed_pancreas_fixture_data()
    generate_larry_fixture_data()
