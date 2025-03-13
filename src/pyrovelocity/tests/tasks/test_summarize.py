"""Tests for `pyrovelocity.tasks.summarize` module."""

import os
import shutil

import pytest

from pyrovelocity.tasks.summarize import summarize_dataset


def test_load_summarize():
    from pyrovelocity.tasks import summarize

    print(summarize.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_summarize_dataset(summarize_dataset_output):
    return summarize_dataset_output


@pytest.mark.slow
@pytest.mark.integration
def test_summarize_dataset_model1(summarize_dataset_model1_output):
    return summarize_dataset_model1_output


@pytest.mark.slow
@pytest.mark.integration
def test_summarize_dataset_pancreas(
    adata_postprocessed_pancreas_50_7,
    pancreas_model2_path,
    pancreas_model2_model_path,
    pancreas_model2_pyrovelocity_data_path,
    tmp_path,
):
    """Test summarizing of the pancreas dataset using real model fixtures.

    This test uses the actual postprocessed data, model, and pyrovelocity data
    from the fixtures to test the summarize_dataset function.
    """
    data_set_name = "pancreas"
    model_identifier = "model2"
    data_model = f"{data_set_name}_{model_identifier}"

    data_model_path = tmp_path / "models" / data_model
    data_model_path.mkdir(parents=True, exist_ok=True)

    postprocessed_data_path = data_model_path / "postprocessed.h5ad"
    adata_postprocessed_pancreas_50_7.write(postprocessed_data_path)

    model_path = data_model_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    for item in pancreas_model2_model_path.iterdir():
        if item.is_file():
            shutil.copy(item, model_path / item.name)

    pyrovelocity_data_path = data_model_path / "pyrovelocity.pkl.zst"
    shutil.copy(pancreas_model2_pyrovelocity_data_path, pyrovelocity_data_path)

    reports_path = tmp_path / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)

    result = summarize_dataset(
        data_model=data_model,
        data_model_path=data_model_path,
        model_path=model_path,
        pyrovelocity_data_path=pyrovelocity_data_path,
        postprocessed_data_path=postprocessed_data_path,
        cell_state="clusters",
        vector_field_basis="umap",
        reports_path=reports_path,
        enable_experimental_plots=False,
        selected_genes=[""],
    )

    reports_dir, dataframe_path = result

    assert os.path.exists(reports_dir)
    assert os.path.isdir(reports_dir)
    assert any(reports_dir.iterdir()), "No reports were generated"

    assert os.path.exists(dataframe_path)
    assert os.path.isfile(dataframe_path)

    expected_files = [
        reports_dir / "shared_time.pdf",
        reports_dir / "volcano.pdf",
        reports_dir / "parameter_uncertainties.pdf",
        reports_dir / "gene_selection_rainbow_plot.pdf",
        reports_dir / "gene_selection_summary_plot.pdf",
        reports_dir / "vector_field.pdf",
    ]

    for file_path in expected_files:
        assert os.path.exists(
            file_path
        ), f"Expected file {file_path} was not generated"

    return result
