import pytest
import scanpy as sc

from pyrovelocity.analysis.analyze import top_mae_genes
from pyrovelocity.io.compressedpickle import CompressedPickle
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


def integration_fixture_scope(fixture_name, config):
    if config.getoption("--cached-integration-fixtures", None):
        return "session"
    return "session"


@pytest.fixture(scope=integration_fixture_scope)
def tmp_data_dir(tmp_path_factory):
    integration_tests_dir = tmp_path_factory.mktemp("integration")
    print(
        f"\nTemporary integration tests directory:\n\n",
        f"{integration_tests_dir}\n",
    )
    return integration_tests_dir


@pytest.fixture(scope=integration_fixture_scope)
def tmp_unit_reports_dir(tmp_data_dir):
    unit_reports_dir = tmp_data_dir / "unit_reports"
    unit_reports_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\nTemporary unit reports directory:\n\n",
        f"{unit_reports_dir}\n",
    )
    return unit_reports_dir


@pytest.fixture(scope=integration_fixture_scope)
def simulated_dataset_path(tmp_data_dir):
    return download_dataset(
        data_set_name="simulated",
        data_external_path=tmp_data_dir / "data/external",
        source="simulate",
        n_obs=100,
        n_vars=200,
    )


@pytest.fixture(scope=integration_fixture_scope)
def preprocess_dataset_output(simulated_dataset_path, tmp_data_dir):
    return preprocess_dataset(
        data_set_name="simulated",
        adata=simulated_dataset_path,
        data_processed_path=tmp_data_dir / "data/processed",
        reports_processed_path=tmp_data_dir / "reports/processed",
    )


@pytest.fixture(scope=integration_fixture_scope)
def train_dataset_output(preprocess_dataset_output, tmp_data_dir):
    _, preprocessed_dataset_path, _ = preprocess_dataset_output
    return train_dataset(
        adata=preprocessed_dataset_path,
        models_path=tmp_data_dir / "models",
        max_epochs=200,
    )


@pytest.fixture(scope=integration_fixture_scope)
def data_model2_reports_path(train_dataset_output, tmp_unit_reports_dir):
    unit_reports_data_model2_path = (
        tmp_unit_reports_dir / train_dataset_output[0]
    )
    unit_reports_data_model2_path.mkdir(parents=True, exist_ok=True)
    print(
        f"\nTemporary data model reports directory:\n\n",
        f"{unit_reports_data_model2_path}\n",
    )
    return unit_reports_data_model2_path


@pytest.fixture(scope=integration_fixture_scope)
def postprocess_dataset_output(train_dataset_output):
    return postprocess_dataset(
        *train_dataset_output[:6],
        vector_field_basis="umap",
        number_posterior_samples=4,
    )


@pytest.fixture(scope=integration_fixture_scope)
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


@pytest.fixture(scope=integration_fixture_scope)
def train_dataset_model1_output(preprocess_dataset_output, tmp_data_dir):
    _, preprocessed_dataset_path, _ = preprocess_dataset_output
    return train_dataset(
        adata=preprocessed_dataset_path,
        model_identifier="model1",
        models_path=tmp_data_dir / "models",
        guide_type="auto_t0_constraint",
        offset=False,
        max_epochs=200,
    )


@pytest.fixture(scope=integration_fixture_scope)
def data_model1_reports_path(train_dataset_model1_output, tmp_unit_reports_dir):
    unit_reports_data_model1_path = (
        tmp_unit_reports_dir / train_dataset_model1_output[0]
    )
    unit_reports_data_model1_path.mkdir(parents=True, exist_ok=True)
    print(
        f"\nTemporary data model 1 unit reports directory:\n\n",
        f"{unit_reports_data_model1_path}\n",
    )
    return unit_reports_data_model1_path


@pytest.fixture(scope=integration_fixture_scope)
def postprocess_dataset_model1_output(train_dataset_model1_output):
    return postprocess_dataset(
        *train_dataset_model1_output[:6],
        vector_field_basis="umap",
        number_posterior_samples=4,
    )


@pytest.fixture(scope=integration_fixture_scope)
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


@pytest.fixture(scope=integration_fixture_scope)
def postprocessed_model2_data(postprocess_dataset_output):
    return sc.read(postprocess_dataset_output[1])


@pytest.fixture(scope=integration_fixture_scope)
def postprocessed_model1_data(postprocess_dataset_model1_output):
    return sc.read(postprocess_dataset_model1_output[1])


@pytest.fixture(scope=integration_fixture_scope)
def posterior_samples_model2(postprocess_dataset_output):
    return CompressedPickle.load(postprocess_dataset_output[0])


@pytest.fixture(scope=integration_fixture_scope)
def putative_model2_marker_genes(posterior_samples_model2):
    return top_mae_genes(
        volcano_data=posterior_samples_model2["gene_ranking"],
    )


@pytest.fixture(scope=integration_fixture_scope)
def posterior_samples_model1(postprocess_dataset_model1_output):
    return CompressedPickle.load(postprocess_dataset_model1_output[0])


@pytest.fixture(scope=integration_fixture_scope)
def putative_model1_marker_genes(posterior_samples_model1):
    return top_mae_genes(
        volcano_data=posterior_samples_model1["gene_ranking"],
    )
