"""Tests for `pyrovelocity.tasks.train` module."""

import pytest

from pyrovelocity.io.serialization import save_anndata_to_json
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.utils import load_anndata_from_path


def test_load_train():
    from pyrovelocity.tasks import train

    print(train.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_train_dataset(train_dataset_output):
    return train_dataset_output


@pytest.mark.slow
@pytest.mark.integration
def test_train_dataset_model1(train_dataset_model1_output):
    return train_dataset_model1_output


def test_train_dataset_pancreas(adata_preprocessed_pancreas_50_7, tmp_path):
    data_set_name = "pancreas"
    model_identifier = "model2"
    models_path = tmp_path / "models"

    result = train_dataset(
        adata=adata_preprocessed_pancreas_50_7,
        data_set_name=data_set_name,
        model_identifier=model_identifier,
        models_path=models_path,
        max_epochs=10,
        force=True,
    )

    (
        data_model,
        data_model_path,
        trained_data_path,
        model_path,
        posterior_samples_path,
        metrics_path,
        run_info_path,
        loss_plot_path,
        loss_csv_path,
    ) = result

    assert data_model == f"{data_set_name}_{model_identifier}"
    assert trained_data_path.exists()
    assert model_path.exists()
    assert posterior_samples_path.exists()
    assert metrics_path.exists()
    assert run_info_path.exists()
    assert loss_plot_path.exists()
    assert loss_csv_path.exists()

    trained_adata = load_anndata_from_path(trained_data_path)
    save_anndata_to_json(trained_adata, tmp_path / "trained_adata.json")
