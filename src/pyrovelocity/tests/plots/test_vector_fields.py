import pytest

from pyrovelocity.plots import plot_vector_field_summary


@pytest.mark.slow
@pytest.mark.integration
def test_model2_plot_vector_field_summary(
    postprocessed_model2_data,
    posterior_samples_model2,
    data_model2_reports_path,
):
    vector_field_summary_plot = (
        data_model2_reports_path / "vector_field_summary_plot.pdf"
    )
    plot_vector_field_summary(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        vector_field_basis="umap",
        plot_name=vector_field_summary_plot,
        cell_state="leiden",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_plot_vector_field_summary(
    postprocessed_model1_data,
    posterior_samples_model1,
    data_model1_reports_path,
):
    vector_field_summary_plot = (
        data_model1_reports_path / "vector_field_summary_plot.pdf"
    )
    plot_vector_field_summary(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        vector_field_basis="umap",
        plot_name=vector_field_summary_plot,
        cell_state="leiden",
    )
