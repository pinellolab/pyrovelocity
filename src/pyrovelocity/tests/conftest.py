import pytest

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
