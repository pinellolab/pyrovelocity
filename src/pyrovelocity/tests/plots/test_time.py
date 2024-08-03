import pytest
import scanpy as sc

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plots import plot_shared_time_uncertainty


@pytest.mark.slow
def test_model2_plot_shared_time_uncertainty(
    postprocess_dataset_output,
    tmp_data_dir,
):
    posterior_samples = CompressedPickle.load(postprocess_dataset_output[0])
    adata = sc.read(postprocess_dataset_output[1])
    shared_time_plot = tmp_data_dir / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=adata,
        posterior_samples=posterior_samples,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )


@pytest.mark.slow
def test_model1_plot_shared_time_uncertainty(
    postprocess_dataset_model1_output,
    tmp_data_dir,
):
    postprocess_dataset_output = postprocess_dataset_model1_output
    posterior_samples = CompressedPickle.load(postprocess_dataset_output[0])
    adata = sc.read(postprocess_dataset_output[1])
    shared_time_plot = tmp_data_dir / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=adata,
        posterior_samples=posterior_samples,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )
