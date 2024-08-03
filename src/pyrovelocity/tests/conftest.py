import pytest

from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.summarize import summarize_dataset
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.utils import generate_sample_data


@pytest.fixture
def default_sample_data():
    """Default sample data from `pyrovelocity.utils.generate_sample_data`.

    Returns:
        AnnData: AnnData object to use for testing.
    """
    return generate_sample_data(random_seed=98)


@pytest.fixture
def default_sample_data_file(default_sample_data, tmp_path):
    """Create a .h5ad file from the default sample data."""
    file_path = tmp_path / "default_sample_data.h5ad"
    default_sample_data.write(file_path)
    return file_path


@pytest.fixture
def tmp_data_dir(tmp_path):
    print(
        f"\nTemporary test data directory:\n\n",
        f"{tmp_path}\n",
    )
    return tmp_path


@pytest.fixture
def simulated_dataset_path(tmp_data_dir):
    return download_dataset(
        data_set_name="simulated",
        data_external_path=tmp_data_dir / "data/external",
        source="simulate",
        n_obs=100,
        n_vars=200,
    )


@pytest.fixture
def preprocess_dataset_output(simulated_dataset_path, tmp_data_dir):
    return preprocess_dataset(
        data_set_name="simulated",
        adata=simulated_dataset_path,
        data_processed_path=tmp_data_dir / "data/processed",
    )


@pytest.fixture
def train_dataset_output(preprocess_dataset_output, tmp_data_dir):
    _, preprocessed_dataset_path = preprocess_dataset_output
    return train_dataset(
        adata=preprocessed_dataset_path,
        models_path=tmp_data_dir / "models",
        max_epochs=200,
    )


@pytest.fixture
def postprocess_dataset_output(train_dataset_output):
    return postprocess_dataset(
        *train_dataset_output[:6],
        vector_field_basis="umap",
        number_posterior_samples=4,
    )


@pytest.fixture
def summarize_dataset_output(
    train_dataset_output,
    postprocess_dataset_output,
    tmp_data_dir,
):
    return summarize_dataset(
        *train_dataset_output[:2],
        model_path=train_dataset_output[3],
        pyrovelocity_data_path=postprocess_dataset_output[0],
        postprocessed_data_path=postprocess_dataset_output[1],
        cell_state="leiden",
        vector_field_basis="umap",
        reports_path=tmp_data_dir / "reports",
    )


@pytest.fixture
def train_dataset_model1_output(preprocess_dataset_output, tmp_data_dir):
    _, preprocessed_dataset_path = preprocess_dataset_output
    return train_dataset(
        adata=preprocessed_dataset_path,
        model_identifier="model1",
        models_path=tmp_data_dir / "models",
        guide_type="auto_t0_constraint",
        offset=False,
        max_epochs=200,
    )


@pytest.fixture
def postprocess_dataset_model1_output(train_dataset_model1_output):
    return postprocess_dataset(
        *train_dataset_model1_output[:6],
        vector_field_basis="umap",
        number_posterior_samples=4,
    )


@pytest.fixture
def summarize_dataset_model1_output(
    train_dataset_model1_output,
    postprocess_dataset_model1_output,
    tmp_data_dir,
):
    return summarize_dataset(
        *train_dataset_model1_output[:2],
        model_path=train_dataset_model1_output[3],
        pyrovelocity_data_path=postprocess_dataset_model1_output[0],
        postprocessed_data_path=postprocess_dataset_model1_output[1],
        cell_state="leiden",
        vector_field_basis="umap",
        reports_path=tmp_data_dir / "reports",
    )
