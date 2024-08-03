import pytest
import scanpy as sc

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plots import plot_vector_field_summary


@pytest.mark.slow
def test_model2_plot_vector_field_summary(
    postprocessed_model2_data,
    posterior_samples_model2,
    tmp_data_dir,
):
    vector_field_summary_plot = tmp_data_dir / "vector_field_summary_plot.pdf"
    plot_vector_field_summary(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        vector_field_basis="umap",
        plot_name=vector_field_summary_plot,
        cell_state="leiden",
    )


@pytest.mark.slow
def test_model1_plot_vector_field_summary(
    postprocessed_model1_data,
    posterior_samples_model1,
    tmp_data_dir,
):
    vector_field_summary_plot = tmp_data_dir / "vector_field_summary_plot.pdf"
    plot_vector_field_summary(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        vector_field_basis="umap",
        plot_name=vector_field_summary_plot,
        cell_state="leiden",
    )
