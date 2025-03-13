import os

import pytest

from pyrovelocity.plots import plot_gene_selection_summary, plot_report


@pytest.mark.slow
@pytest.mark.integration
def test_model2_plot_gene_selection_summary(
    postprocessed_model2_data,
    posterior_samples_model2,
    putative_model2_marker_genes,
    data_model2_reports_path,
):
    plot_gene_selection_summary(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        basis="umap",
        cell_state="leiden",
        plot_name=data_model2_reports_path / "gene_selection_summary_plot.pdf",
        selected_genes=putative_model2_marker_genes,
        show_marginal_histograms=False,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_plot_gene_selection_summary(
    postprocessed_model1_data,
    posterior_samples_model1,
    putative_model1_marker_genes,
    data_model1_reports_path,
):
    plot_gene_selection_summary(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        basis="umap",
        cell_state="leiden",
        plot_name=data_model1_reports_path / "gene_selection_summary_plot.pdf",
        selected_genes=putative_model1_marker_genes,
        show_marginal_histograms=False,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_plot_report(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    tmp_path,
):
    """Test plot_report function with pancreas dataset fixtures."""
    output_path = tmp_path / "gene_selection_summary_plot.pdf"
    figure_path = f"{output_path}.dill.zst"

    plot_report(
        adata=adata_postprocessed_pancreas_50_7,
        posterior_samples=pancreas_model2_pyrovelocity_data,
        volcano_data=pancreas_model2_pyrovelocity_data["gene_ranking"],
        putative_marker_genes=pancreas_model2_putative_marker_genes,
        selected_genes=[""],
        vector_field_basis="umap",
        cell_state="clusters",
        state_color_dict=None,
        report_file_path=output_path,
        figure_file_path=figure_path,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    assert os.path.exists(figure_path)
    assert os.path.getsize(figure_path) > 0


# def test_plot_report_with_selected_genes(
#     adata_postprocessed_pancreas_50_7,
#     pancreas_model2_pyrovelocity_data,
#     pancreas_model2_putative_marker_genes,
#     tmp_path,
# ):
#     """Test plot_report function with specific selected genes."""
#     output_path = tmp_path / "gene_selection_summary_selected.pdf"
#     figure_path = f"{output_path}.dill.zst"

#     selected_genes = list(adata_postprocessed_pancreas_50_7.var_names[:2])

#     plot_report(
#         adata=adata_postprocessed_pancreas_50_7,
#         posterior_samples=pancreas_model2_pyrovelocity_data,
#         volcano_data=pancreas_model2_pyrovelocity_data["gene_ranking"],
#         putative_marker_genes=pancreas_model2_putative_marker_genes,
#         selected_genes=selected_genes,
#         vector_field_basis="umap",
#         cell_state="clusters",
#         state_color_dict=None,
#         report_file_path=output_path,
#         figure_file_path=figure_path,
#     )

#     assert os.path.exists(output_path)
#     assert os.path.getsize(output_path) > 0
#     assert os.path.exists(figure_path)
#     assert os.path.getsize(figure_path) > 0
