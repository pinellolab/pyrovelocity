import os
import tempfile
import uuid
from importlib.resources import files

import pytest
import scanpy as sc

from pyrovelocity.analysis.analyze import top_mae_genes
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.summarize import summarize_dataset
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.utils import generate_sample_data


@pytest.fixture
def adata_preprocessed_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(fixture_file_path)


@pytest.fixture
def adata_trained_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "trained_pancreas_50_7.json"
    )
    return load_anndata_from_json(fixture_file_path)


@pytest.fixture
def adata_postprocessed_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "postprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(fixture_file_path)


@pytest.fixture
def adata_larry_multilineage_50_6():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "larry_multilineage_50_6.json"
    )
    return load_anndata_from_json(fixture_file_path)


@pytest.fixture
def pancreas_model2_path():
    return files("pyrovelocity.tests.data") / "models" / "pancreas_model2"


@pytest.fixture
def pancreas_model2_posterior_samples_path(pancreas_model2_path):
    return pancreas_model2_path / "posterior_samples.pkl.zst"


@pytest.fixture
def pancreas_model2_pyrovelocity_data_path(pancreas_model2_path):
    return pancreas_model2_path / "pyrovelocity.pkl.zst"


@pytest.fixture
def pancreas_model2_pyrovelocity_data(pancreas_model2_pyrovelocity_data_path):
    return CompressedPickle.load(pancreas_model2_pyrovelocity_data_path)


@pytest.fixture
def pancreas_model2_putative_marker_genes(pancreas_model2_pyrovelocity_data):
    return top_mae_genes(
        volcano_data=pancreas_model2_pyrovelocity_data["gene_ranking"],
    )


@pytest.fixture
def pancreas_model2_model_path(pancreas_model2_path):
    return pancreas_model2_path / "model"


@pytest.fixture
def pancreas_model2_metrics_path(pancreas_model2_path):
    return pancreas_model2_path / "metrics.json"


@pytest.fixture
def pancreas_model2_run_info_path(pancreas_model2_path):
    return pancreas_model2_path / "run_info.json"


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


# General-purpose fixtures for temporary file handling
@pytest.fixture
def temp_file_path():
    """Create a unique temporary file path for each test.

    This fixture provides a unique file path (not an actual file) for tests
    that need to write and read files. The path is automatically cleaned up
    after the test completes.

    Returns:
        str: A unique temporary file path
    """
    # Create a unique filename using uuid to avoid conflicts in parallel testing
    unique_filename = f"test_data_{uuid.uuid4().hex}"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, unique_filename)

    yield file_path

    # Clean up after the test
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture
def temp_compressed_pickle_path():
    """Create a unique temporary file path for compressed pickle files.

    This fixture provides a unique file path with the .pkl.zst extension
    for tests that need to work with CompressedPickle. The path is
    automatically cleaned up after the test completes.

    Returns:
        str: A unique temporary file path with .pkl.zst extension
    """
    # Create a unique filename using uuid to avoid conflicts in parallel testing
    unique_filename = f"test_data_{uuid.uuid4().hex}.pkl.zst"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, unique_filename)

    yield file_path

    # Clean up after the test
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture
def save_and_load_helper():
    """Helper function to save and load data with CompressedPickle.

    This fixture provides a function that handles both saving and loading
    data with CompressedPickle, making tests more concise and reliable.

    Returns:
        function: A function that saves and loads data
    """

    def _save_and_load(data, file_path, **kwargs):
        """Save data to a file and load it back.

        Args:
            data: The data to save
            file_path: The path to save the data to
            **kwargs: Additional arguments to pass to CompressedPickle.save
                      and CompressedPickle.load

        Returns:
            The loaded data
        """
        save_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["sparsify", "density_threshold"]
        }
        load_kwargs = {k: v for k, v in kwargs.items() if k in ["densify"]}

        CompressedPickle.save(file_path=file_path, obj=data, **save_kwargs)
        return CompressedPickle.load(file_path=file_path, **load_kwargs)

    return _save_and_load
