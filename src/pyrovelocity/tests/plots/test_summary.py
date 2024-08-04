import pytest

from pyrovelocity.plots import plot_gene_selection_summary


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
