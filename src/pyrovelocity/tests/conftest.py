import pytest

from pyrovelocity.utils import generate_sample_data


@pytest.fixture
def default_sample_data():
    """Default sample data from `pyrovelocity.utils.generate_sample_data`.

    Returns:
        AnnData: AnnData object to use for testing.
    """
    return generate_sample_data(random_seed=98)
