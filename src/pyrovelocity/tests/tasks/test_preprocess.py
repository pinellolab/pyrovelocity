"""Tests for `pyrovelocity.tasks.preprocess` module."""
import pytest

from pyrovelocity.tasks.preprocess import preprocess_dataset


def test_load_preprocess():
    from pyrovelocity.tasks import preprocess

    print(preprocess.__file__)


@pytest.mark.slow
def test_preprocess_dataset(simulated_dataset_path, tmp_tasks_dir):
    preprocess_dataset(
        data_set_name="simulated",
        adata=simulated_dataset_path,
        data_processed_path=tmp_tasks_dir / "data/processed",
    )
