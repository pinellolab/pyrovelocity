import pytest

from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.train import train_dataset


@pytest.fixture
def tmp_tasks_dir(tmp_path):
    print(
        f"\nTemporary test data directory:\n\n",
        f"{tmp_path}\n",
    )
    return tmp_path


@pytest.fixture
def simulated_dataset_path(tmp_tasks_dir):
    return download_dataset(
        data_set_name="simulated",
        data_external_path=tmp_tasks_dir / "data/external",
        source="simulate",
        n_obs=100,
        n_vars=200,
    )


@pytest.fixture
def preprocess_dataset_output(simulated_dataset_path, tmp_tasks_dir):
    return preprocess_dataset(
        data_set_name="simulated",
        adata=simulated_dataset_path,
        data_processed_path=tmp_tasks_dir / "data/processed",
    )


@pytest.fixture
def train_dataset_output(preprocess_dataset_output, tmp_tasks_dir):
    _, preprocessed_dataset_path = preprocess_dataset_output
    return train_dataset(
        adata=preprocessed_dataset_path,
        models_path=tmp_tasks_dir / "models",
        max_epochs=200,
    )


@pytest.fixture
def postprocess_dataset_output(train_dataset_output, tmp_tasks_dir):
    return postprocess_dataset(
        *train_dataset_output[:6],
        vector_field_basis="umap",
        number_posterior_samples=4,
    )


@pytest.fixture
def summarize_dataset_output(
    train_dataset_output,
    postprocess_dataset_output,
    tmp_tasks_dir,
):
    return summarize_dataset(
        *train_dataset_output[:2],
        model_path=train_dataset_output[3],
        pyrovelocity_data_path=postprocess_dataset_output[0],
        postprocessed_data_path=postprocess_dataset_output[1],
        cell_state="leiden",
        vector_field_basis="umap",
        reports_path=tmp_tasks_dir / "reports",
    )

    )
