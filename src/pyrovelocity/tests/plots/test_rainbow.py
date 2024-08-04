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
