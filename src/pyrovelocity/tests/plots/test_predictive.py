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
