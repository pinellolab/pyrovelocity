import pytest

from pyrovelocity.plots import posterior_curve


@pytest.mark.slow
@pytest.mark.integration
def test_model2_plot_posterior_predictive(
    postprocessed_model2_data,
    posterior_samples_model2,
    putative_model2_marker_genes,
    train_dataset_output,
    data_model2_reports_path,
):
    posterior_curve(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        gene_set=putative_model2_marker_genes,
        data_model=train_dataset_output[0],
        model_path=train_dataset_output[3],
        output_directory=data_model2_reports_path / "posterior_phase_portraits",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_plot_posterior_predictive(
    postprocessed_model1_data,
    posterior_samples_model1,
    putative_model1_marker_genes,
    train_dataset_model1_output,
    data_model1_reports_path,
):
    posterior_curve(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        gene_set=putative_model1_marker_genes,
        data_model=train_dataset_model1_output[0],
        model_path=train_dataset_model1_output[3],
        output_directory=data_model1_reports_path / "posterior_phase_portraits",
    )


def test_posterior_curve(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    pancreas_model2_model_path,
    tmp_path,
):
    """Test posterior_curve function with pancreas dataset fixtures."""
    output_directory = tmp_path / "posterior_phase_portraits"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Use a subset of genes to speed up the test
    gene_subset = (
        pancreas_model2_putative_marker_genes[:1]
        if pancreas_model2_putative_marker_genes
        else None
    )

    if gene_subset:
        posterior_curve(
            adata=adata_postprocessed_pancreas_50_7,
            posterior_samples=pancreas_model2_pyrovelocity_data,
            gene_set=gene_subset,
            data_model="pancreas_model2",
            model_path=pancreas_model2_model_path,
            output_directory=output_directory,
        )

        # Check if any files were created in the output directory
        assert any(
            output_directory.iterdir()
        ), "No files were generated in the output directory"
    else:
        pytest.skip("No marker genes available for posterior curve testing")


def test_posterior_curve_multiple_genes(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    pancreas_model2_putative_marker_genes,
    pancreas_model2_model_path,
    tmp_path,
):
    """Test posterior_curve function with multiple genes."""
    output_directory = tmp_path / "posterior_phase_portraits_multiple"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Use a subset of genes to speed up the test
    gene_subset = (
        pancreas_model2_putative_marker_genes[:2]
        if len(pancreas_model2_putative_marker_genes) >= 2
        else None
    )

    if gene_subset and len(gene_subset) >= 2:
        posterior_curve(
            adata=adata_postprocessed_pancreas_50_7,
            posterior_samples=pancreas_model2_pyrovelocity_data,
            gene_set=gene_subset,
            data_model="pancreas_model2",
            model_path=pancreas_model2_model_path,
            output_directory=output_directory,
        )

        # Check if any files were created in the output directory
        assert any(
            output_directory.iterdir()
        ), "No files were generated in the output directory"
    else:
        pytest.skip(
            "Not enough marker genes available for multiple gene testing"
        )


# def test_posterior_curve_with_options(
#     adata_postprocessed_pancreas_50_7,
#     pancreas_model2_pyrovelocity_data,
#     pancreas_model2_putative_marker_genes,
#     pancreas_model2_model_path,
#     tmp_path,
# ):
#     """Test posterior_curve function with additional options if supported."""
#     output_directory = tmp_path / "posterior_phase_portraits_options"
#     output_directory.mkdir(parents=True, exist_ok=True)

#     # Use a subset of genes to speed up the test
#     gene_subset = (
#         pancreas_model2_putative_marker_genes[:1]
#         if pancreas_model2_putative_marker_genes
#         else None
#     )

#     if gene_subset:
#         # Try with additional parameters if supported
#         try:
#             posterior_curve(
#                 adata=adata_postprocessed_pancreas_50_7,
#                 posterior_samples=pancreas_model2_pyrovelocity_data,
#                 gene_set=gene_subset,
#                 data_model="pancreas_model2",
#                 model_path=pancreas_model2_model_path,
#                 output_directory=output_directory,
#                 n_samples=5,  # Additional parameter if supported
#                 figsize=(10, 10),  # Additional parameter if supported
#                 dpi=300,  # Additional parameter if supported
#             )

#             # Check if any files were created in the output directory
#             assert any(
#                 output_directory.iterdir()
#             ), "No files were generated in the output directory"
#         except TypeError:
#             # If the function doesn't accept these parameters, skip this test
#             pytest.skip("Function doesn't support additional parameters")
#     else:
#         pytest.skip("No marker genes available for posterior curve testing")
