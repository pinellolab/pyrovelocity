import pytest
import scanpy as sc

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plots import plot_vector_field_summary


@pytest.mark.slow
def test_model2_plot_vector_field_summary(
    postprocess_dataset_output,
    tmp_data_dir,
):
    posterior_samples = CompressedPickle.load(postprocess_dataset_output[0])
    adata = sc.read(postprocess_dataset_output[1])
    vector_field_summary_plot = tmp_data_dir / "vector_field_summary_plot.pdf"
    plot_vector_field_summary(
        adata=adata,
        posterior_samples=posterior_samples,
        vector_field_basis="umap",
        plot_name=vector_field_summary_plot,
        cell_state="leiden",
    )


@pytest.mark.slow
def test_model1_plot_vector_field_summary(
    postprocess_dataset_model1_output,
    tmp_data_dir,
):
    postprocess_dataset_output = postprocess_dataset_model1_output
    posterior_samples = CompressedPickle.load(postprocess_dataset_output[0])
    adata = sc.read(postprocess_dataset_output[1])
    vector_field_summary_plot = tmp_data_dir / "vector_field_summary_plot.pdf"
    plot_vector_field_summary(
        adata=adata,
        posterior_samples=posterior_samples,
        vector_field_basis="umap",
        plot_name=vector_field_summary_plot,
        cell_state="leiden",
    )
