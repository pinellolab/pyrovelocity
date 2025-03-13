import os

import pytest

from pyrovelocity.plots import rainbowplot


@pytest.mark.slow
@pytest.mark.integration
def test_model2_rainbowplot(
    postprocessed_model2_data,
    posterior_samples_model2,
    putative_model2_marker_genes,
    data_model2_reports_path,
):
    rainbowplot(
        volcano_data=posterior_samples_model2["gene_ranking"],
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        genes=putative_model2_marker_genes,
        data=["st", "ut"],
        basis="umap",
        cell_state="leiden",
        save_plot=True,
        rainbow_plot_path=data_model2_reports_path
        / "gene_selection_rainbow_plot.pdf",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_rainbowplot(
    postprocessed_model1_data,
    posterior_samples_model1,
    putative_model1_marker_genes,
    data_model1_reports_path,
):
    rainbowplot(
        volcano_data=posterior_samples_model1["gene_ranking"],
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        genes=putative_model1_marker_genes,
        data=["st", "ut"],
        basis="umap",
        cell_state="leiden",
        save_plot=True,
        rainbow_plot_path=data_model1_reports_path
        / "gene_selection_rainbow_plot.pdf",
    )


def test_rainbowplot(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    tmp_path,
):
    """Test rainbowplot function with pancreas dataset fixtures."""
    output_path = tmp_path / "gene_selection_rainbow_plot.pdf"

    fig = rainbowplot(
        volcano_data=pancreas_model2_pyrovelocity_data["gene_ranking"],
        adata=adata_postprocessed_pancreas_50_7,
        posterior_samples=pancreas_model2_pyrovelocity_data,
        genes=pancreas_model2_putative_marker_genes,
        data=["st", "ut"],
        basis="umap",
        cell_state="clusters",
        save_plot=True,
        rainbow_plot_path=output_path,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    assert fig is not None


def test_rainbowplot_no_save(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
):
    """Test rainbowplot function without saving."""
    fig = rainbowplot(
        volcano_data=pancreas_model2_pyrovelocity_data["gene_ranking"],
        adata=adata_postprocessed_pancreas_50_7,
        posterior_samples=pancreas_model2_pyrovelocity_data,
        genes=pancreas_model2_putative_marker_genes,
        data=["st", "ut"],
        basis="umap",
        cell_state="clusters",
        save_plot=False,
    )

    assert fig is not None
