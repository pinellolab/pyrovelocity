import os

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


def test_plot_parameter_posterior_distributions(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    tmp_path,
):
    """Test plot_parameter_posterior_distributions function with pancreas dataset fixtures."""
    output_path = tmp_path / "parameter_uncertainties.pdf"

    plot_parameter_posterior_distributions(
        posterior_samples=pancreas_model2_pyrovelocity_data,
        adata=adata_postprocessed_pancreas_50_7,
        geneset=pancreas_model2_putative_marker_genes,
        save_plot=True,
        parameter_uncertainty_plot=output_path,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_plot_parameter_posterior_distributions_no_save(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
):
    """Test plot_parameter_posterior_distributions function without saving."""
    fig = plot_parameter_posterior_distributions(
        posterior_samples=pancreas_model2_pyrovelocity_data,
        adata=adata_postprocessed_pancreas_50_7,
        geneset=pancreas_model2_putative_marker_genes,
        save_plot=False,
    )

    assert fig is not None


# def test_plot_parameter_posterior_distributions_specific_genes(
#     adata_postprocessed_pancreas_50_7,
#     pancreas_model2_pyrovelocity_data,
#     tmp_path,
# ):
#     """Test plot_parameter_posterior_distributions function with specific genes."""
#     output_path = tmp_path / "parameter_uncertainties_specific.pdf"

#     specific_genes = list(adata_postprocessed_pancreas_50_7.var_names[:2])

#     plot_parameter_posterior_distributions(
#         posterior_samples=pancreas_model2_pyrovelocity_data,
#         adata=adata_postprocessed_pancreas_50_7,
#         geneset=specific_genes,
#         save_plot=True,
#         parameter_uncertainty_plot=output_path,
#     )

#     assert os.path.exists(output_path)
#     assert os.path.getsize(output_path) > 0
