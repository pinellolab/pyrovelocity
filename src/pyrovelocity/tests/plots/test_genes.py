import pytest

from pyrovelocity.plots import plot_gene_ranking


@pytest.mark.slow
@pytest.mark.integration
def test_model2_plot_gene_ranking(
    postprocessed_model2_data,
    posterior_samples_model2,
    putative_model2_marker_genes,
    data_model2_reports_path,
):
    plot_gene_ranking(
        posterior_samples=[posterior_samples_model2],
        adata=[postprocessed_model2_data],
        selected_genes=putative_model2_marker_genes,
        time_correlation_with="st",
        show_marginal_histograms=True,
        save_volcano_plot=True,
        volcano_plot_path=data_model2_reports_path / "volcano.pdf",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_plot_gene_ranking(
    postprocessed_model1_data,
    posterior_samples_model1,
    putative_model1_marker_genes,
    data_model1_reports_path,
):
    plot_gene_ranking(
        posterior_samples=[posterior_samples_model1],
        adata=[postprocessed_model1_data],
        selected_genes=putative_model1_marker_genes,
        time_correlation_with="st",
        show_marginal_histograms=True,
        save_volcano_plot=True,
        volcano_plot_path=data_model1_reports_path / "volcano.pdf",
    )
