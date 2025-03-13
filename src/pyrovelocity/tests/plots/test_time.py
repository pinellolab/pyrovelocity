import os
from pathlib import Path

import pytest

from pyrovelocity.plots import plot_shared_time_uncertainty


@pytest.mark.slow
@pytest.mark.integration
def test_model2_plot_shared_time_uncertainty(
    postprocessed_model2_data,
    posterior_samples_model2,
    data_model2_reports_path,
):
    shared_time_plot = data_model2_reports_path / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=postprocessed_model2_data,
        posterior_samples=posterior_samples_model2,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_model1_plot_shared_time_uncertainty(
    postprocessed_model1_data,
    posterior_samples_model1,
    data_model1_reports_path,
):
    shared_time_plot = data_model1_reports_path / "shared_time_plot.pdf"
    plot_shared_time_uncertainty(
        adata=postprocessed_model1_data,
        posterior_samples=posterior_samples_model1,
        vector_field_basis="umap",
        shared_time_plot=shared_time_plot,
    )


def test_plot_shared_time_uncertainty(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_pyrovelocity_data,
    tmp_path,
):
    """Test plot_shared_time_uncertainty function with pancreas dataset fixtures."""
    output_path = tmp_path / "shared_time.pdf"

    plot_shared_time_uncertainty(
        adata=adata_postprocessed_pancreas_50_7,
        posterior_samples=pancreas_model2_pyrovelocity_data,
        vector_field_basis="umap",
        shared_time_plot=output_path,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
