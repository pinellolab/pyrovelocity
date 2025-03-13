import os

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
        posterior_samples=posterior_samples_model2,
        adata=postprocessed_model2_data,
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
        posterior_samples=posterior_samples_model1,
        adata=postprocessed_model1_data,
        selected_genes=putative_model1_marker_genes,
        time_correlation_with="st",
        show_marginal_histograms=True,
        save_volcano_plot=True,
        volcano_plot_path=data_model1_reports_path / "volcano.pdf",
    )


def test_plot_gene_ranking(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    tmp_path,
):
    """Test plot_gene_ranking function with pancreas dataset fixtures."""
    output_path = tmp_path / "volcano.pdf"

    plot_gene_ranking(
        posterior_samples=pancreas_model2_pyrovelocity_data,
        adata=adata_postprocessed_pancreas_50_7,
        putative_marker_genes=pancreas_model2_putative_marker_genes,
        selected_genes=[""],
        time_correlation_with="st",
        show_marginal_histograms=True,
        save_volcano_plot=True,
        volcano_plot_path=output_path,
        show_xy_labels=True,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_plot_gene_ranking_with_selected_genes(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    tmp_path,
):
    """Test plot_gene_ranking function with specific selected genes."""
    output_path = tmp_path / "volcano_selected.pdf"

    # Get a few gene names from the dataset to use as selected genes
    selected_genes = list(adata_postprocessed_pancreas_50_7.var_names[:2])

    plot_gene_ranking(
        posterior_samples=pancreas_model2_pyrovelocity_data,
        adata=adata_postprocessed_pancreas_50_7,
        putative_marker_genes=pancreas_model2_putative_marker_genes,
        selected_genes=selected_genes,
        time_correlation_with="st",
        show_marginal_histograms=True,
        save_volcano_plot=True,
        volcano_plot_path=output_path,
        show_xy_labels=True,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_plot_gene_ranking_without_histograms(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    tmp_path,
):
    """Test plot_gene_ranking function without marginal histograms."""
    output_path = tmp_path / "volcano_no_histograms.pdf"

    plot_gene_ranking(
        posterior_samples=pancreas_model2_pyrovelocity_data,
        adata=adata_postprocessed_pancreas_50_7,
        putative_marker_genes=pancreas_model2_putative_marker_genes,
        selected_genes=[""],
        time_correlation_with="st",
        show_marginal_histograms=False,
        save_volcano_plot=True,
        volcano_plot_path=output_path,
        show_xy_labels=True,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
