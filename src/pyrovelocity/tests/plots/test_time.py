import pytest

from pyrovelocity.plots import plot_shared_time_uncertainty


@pytest.mark.slow
def test_model2_plot_shared_time_uncertainty(
    postprocessed_model2_data,
    posterior_samples_model2,
    tmp_data_dir,
):
    shared_time_plot = tmp_data_dir / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )


@pytest.mark.slow
def test_model1_plot_shared_time_uncertainty(
    postprocessed_model1_data,
    posterior_samples_model1,
    tmp_data_dir,
):
    shared_time_plot = tmp_data_dir / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )
