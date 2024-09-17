"""
Execute this script to generate test fixture data:

python src/pyrovelocity/tests/fixtures/generate_anndata_fixtures.py

This script should not be executed by pytest.
It is used to generate test fixture data.
"""

from pathlib import Path

from anndata import AnnData

from pyrovelocity.io.datasets import pancreas
from pyrovelocity.io.serialization import (
    save_anndata_to_json,
)
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.utils import configure_logging, print_anndata
from pyrovelocity.workflows.main_configuration import (
    pancreas_summary_configuration,
)

__all__ = ["generate_pancreas_fixture_data"]

logger = configure_logging(__name__)


def generate_pancreas_fixture_data(
    output_path: str
    | Path = "src/pyrovelocity/tests/data/preprocessed_pancreas.json",
    n_obs: int = 50,
    n_vars: int = 7,
) -> Path:
    """
    Generate a test fixture for the pancreas dataset.

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

    selected_genes = pancreas_summary_configuration.selected_genes
    selected_genes_in_adata = [
        gene for gene in selected_genes if gene in adata.var_names
    ]
    if len(selected_genes_in_adata) < len(selected_genes):
        logger.warning(
            f"Warning: Some selected genes are not in the downsampled data:",
            f"{set(selected_genes) - set(selected_genes_in_adata)}",
        )

    genes_to_keep = list(
        set(adata.var_names[:n_vars].tolist() + selected_genes_in_adata)
    )
    adata = adata[:, genes_to_keep]

    print_anndata(adata)
    save_anndata_to_json(adata, output_path)

    logger.info(f"Test fixture saved to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_pancreas_fixture_data()
