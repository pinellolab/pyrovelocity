"""Tests for `pyrovelocity._train_model` task."""

from pyrovelocity.tasks.preprocess import copy_raw_counts
from pyrovelocity.tasks.train import train_model
from pyrovelocity.utils import generate_sample_data


def test_train_model(tmp_path):
    loss_plot_path = str(tmp_path) + "/loss_plot_docs.png"
    print(loss_plot_path)
    adata = generate_sample_data(random_seed=99)
    copy_raw_counts(adata)
    _, model, posterior_samples = train_model(
        adata,
        adata_atac=None,
        use_gpu="auto",
        seed=99,
        max_epochs=200,
        loss_plot_path=loss_plot_path,
    )
