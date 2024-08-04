import pytest

from pyrovelocity.plots import plot_shared_time_uncertainty


@pytest.mark.slow
@pytest.mark.integration
def test_model2_plot_shared_time_uncertainty(
    postprocessed_model2_data,
    posterior_samples_model2,
    data_model2_reports_path,
):
    shared_time_plot = data_model2_reports_path / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_plot_shared_time_uncertainty(
    postprocessed_model1_data,
    posterior_samples_model1,
    data_model1_reports_path,
):
    shared_time_plot = data_model1_reports_path / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )
