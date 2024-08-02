import pytest

from pyrovelocity.tasks.data import download_dataset


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
