import pytest

from pyrovelocity.plots import plot_parameter_posterior_distributions


@pytest.mark.slow
@pytest.mark.integration
def test_model2_plot_parameter_posterior_distributions(
    postprocessed_model2_data,
    posterior_samples_model2,
    putative_model2_marker_genes,
    data_model2_reports_path,
):
    plot_parameter_posterior_distributions(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        geneset=putative_model2_marker_genes,
        parameter_uncertainty_plot=data_model2_reports_path
        / "parameter_uncertainties.pdf",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_plot_parameter_posterior_distributions(
    postprocessed_model1_data,
    posterior_samples_model1,
    putative_model1_marker_genes,
    data_model1_reports_path,
):
    plot_parameter_posterior_distributions(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        geneset=putative_model1_marker_genes,
        parameter_uncertainty_plot=data_model1_reports_path
        / "parameter_uncertainties.pdf",
    )
