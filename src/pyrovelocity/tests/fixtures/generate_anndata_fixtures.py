"""
Execute this script to generate test fixture data:

python src/pyrovelocity/tests/fixtures/generate_anndata_fixtures.py

This script should not be executed by pytest.
It is used to generate test fixture data.
"""

from pathlib import Path

from anndata import AnnData
from beartype import beartype
from beartype.typing import List, Union

from pyrovelocity.io.datasets import pancreas
from pyrovelocity.io.serialization import (
    load_anndata_from_json,
    save_anndata_to_json,
)
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.utils import (
    anndata_string,
    configure_logging,
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


if __name__ == "__main__":
    generate_pancreas_fixture_data()
