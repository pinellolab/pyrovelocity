"""Model training API for Pyro-Velocity."""
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from anndata._core.anndata import AnnData
from numpy import ndarray
from pyro import poutine
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.guides import AutoGuideList
from sklearn.model_selection import train_test_split

from pyrovelocity._velocity import PyroVelocity


def train_model(
    adata: AnnData,
    guide_type: str = "auto",
    model_type: str = "auto",
    svi_train: bool = False,  # svi_train alreadys turn off
    batch_size: int = -1,
    train_size: float = 1.0,
    use_gpu: int = 0,
    likelihood: str = "Poisson",
    num_samples: int = 30,
    log_every: int = 100,
    cell_state: str = "clusters",
    patient_improve: float = 5e-4,
    patient_init: int = 30,
    seed: int = 99,
    lr: float = 0.01,
    max_epochs: int = 3000,
    include_prior: bool = True,
    library_size: bool = True,
    offset: bool = False,
    input_type: str = "raw",
    cell_specific_kinetics: Optional[str] = None,
    kinetics_num: int = 2,
    loss_plot_path: str = "loss_plot.png",
) -> Tuple[PyroVelocity, Dict[str, ndarray]]:
    """
    Train a PyroVelocity model to provide probabilistic estimates of RNA velocity
    for single-cell RNA sequencing data with quantified splice variants.

    Args:
        adata (AnnData): An AnnData object containing the input data.
        guide_type (str, optional): The type of guide function for the Pyro model. Default is "auto".
        model_type (str, optional): The type of Pyro model. Default is "auto".
        svi_train (bool, optional): Whether to use Stochastic Variational Inference for training. Default is False.
        batch_size (int, optional): Batch size for training. Default is -1, which indicates using the full dataset.
        train_size (float, optional): Proportion of data to be used for training. Default is 1.0.
        use_gpu (int, optional): Whether to use GPU for training. Default is 0, which indicates not using GPU.
        likelihood (str, optional): Likelihood function for the Pyro model. Default is "Poisson".
        num_samples (int, optional): Number of posterior samples. Default is 30.
        log_every (int, optional): Frequency of logging progress. Default is 100.
        cell_state (str, optional): Cell state attribute in the AnnData object. Default is "clusters".
        patient_improve (float, optional): Minimum improvement in training loss for early stopping. Default is 5e-4.
        patient_init (int, optional): Number of initial training epochs before early stopping is enabled. Default is 30.
        seed (int, optional): Random seed for reproducibility. Default is 99.
        lr (float, optional): Learning rate for the optimizer. Default is 0.01.
        max_epochs (int, optional): Maximum number of training epochs. Default is 3000.
        include_prior (bool, optional): Whether to include prior information in the model. Default is True.
        library_size (bool, optional): Whether to correct for library size. Default is True.
        offset (bool, optional): Whether to add an offset to the model. Default is False.
        input_type (str, optional): Type of input data. Default is "raw".
        cell_specific_kinetics (Optional[str], optional): Name of the attribute containing cell-specific kinetics information. Default is None.
        kinetics_num (int, optional): Number of kinetics parameters. Default is 2.
        loss_plot_path (str, optional): Path to save the loss plot. Default is "loss_plot.png".

    Returns:
        Tuple[PyroVelocity, Dict[str, ndarray]]: A tuple containing the trained PyroVelocity model and a dictionary of posterior samples.

    Examples:
        >>> from pyrovelocity.api import train_model
        >>> from pyrovelocity.utils import generate_sample_data
        >>> from pyrovelocity.data import copy_raw_counts
        >>> adata = generate_sample_data(random_seed=99)
        >>> copy_raw_counts(adata)
        >>> model, posterior_samples = train_model(adata, seed=99, max_epochs=200, loss_plot_path="loss_plot_docs.png")
    """
    PyroVelocity.setup_anndata(adata)

    model = PyroVelocity(
        adata,
        likelihood=likelihood,
        model_type=model_type,
        guide_type=guide_type,
        correct_library_size=library_size,
        add_offset=offset,
        include_prior=include_prior,
        input_type=input_type,
        cell_specific_kinetics=cell_specific_kinetics,
        kinetics_num=kinetics_num,
    )
    if svi_train and guide_type in {
        "velocity_auto",
        "velocity_auto_t0_constraint",
    }:
        if batch_size == -1:
            batch_size = adata.shape[0]
        model.train(
            max_epochs=max_epochs,
            lr=lr,
            use_gpu=use_gpu,
            batch_size=batch_size,
            train_size=train_size,
            valid_size=1 - train_size,
            check_val_every_n_epoch=1,
            early_stopping=True,
            patience=patient_init,
            min_delta=patient_improve,
        )

        fig, ax = plt.subplots()
        fig.set_size_inches(2.5, 1.5)
        ax.scatter(
            model.history_["elbo_train"].index[:-1],
            -model.history_["elbo_train"][:-1],
            label="Train",
        )
        if train_size < 1:
            ax.scatter(
                model.history_["elbo_validation"].index[:-1],
                -model.history_["elbo_validation"][:-1],
                label="Valid",
            )
        set_loss_plot_axes(ax)
        fig.savefig(loss_plot_path, facecolor="white", bbox_inches="tight")
        posterior_samples = model.generate_posterior_samples(
            model.adata, num_samples=num_samples, batch_size=512
        )
        return model, posterior_samples
    else:
        if train_size >= 1:  ##support velocity_auto_depth
            if batch_size == -1:
                batch_size = adata.shape[0]

            if batch_size >= adata.shape[0]:
                losses = model.train_faster(
                    max_epochs=max_epochs,
                    lr=lr,
                    use_gpu=use_gpu,
                    seed=seed,
                    patient_improve=patient_improve,
                    patient_init=patient_init,
                    log_every=log_every,
                )
            else:
                losses = model.train_faster_with_batch(
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    log_every=log_every,
                    lr=lr,
                    use_gpu=use_gpu,
                    seed=seed,
                    patient_improve=patient_improve,
                    patient_init=patient_init,
                )
            fig, ax = plt.subplots()
            fig.set_size_inches(2.5, 1.5)
            ax.scatter(
                np.arange(len(losses)),
                -np.array(losses),
                label="train",
                alpha=0.25,
            )
            set_loss_plot_axes(ax)
            posterior_samples = model.generate_posterior_samples(
                model.adata, num_samples=num_samples, batch_size=512
            )

            fig.savefig(loss_plot_path, facecolor="white", bbox_inches="tight")

            return model, posterior_samples
        else:  # train validation procedure
            if (
                guide_type == "velocity_auto_depth"
            ):  # velocity_auto, not supported (velocity_auto_depth, fails with error)
                raise

            indices = np.arange(adata.shape[0])
            train_ind, test_ind, cluster_train, cluster_test = train_test_split(
                indices,
                adata.obs.loc[:, cell_state].values,
                test_size=1 - train_size,
                random_state=seed,
                shuffle=False,
            )
            train_batch_size = (
                train_ind.shape[0] if batch_size == -1 else batch_size
            )
            losses = model.train_faster_with_batch(
                max_epochs=max_epochs,
                batch_size=train_batch_size,
                indices=train_ind,
                log_every=log_every,
                lr=lr,
                use_gpu=use_gpu,
                seed=seed,
                patient_improve=patient_improve,
                patient_init=patient_init,
            )
            posterior_samples = model.generate_posterior_samples(
                model.adata,
                num_samples=num_samples,
                indices=train_ind,
                batch_size=512,
            )

            test_batch_size = (
                test_ind.shape[0] if batch_size == -1 else batch_size
            )
            if guide_type in {"auto", "auto_t0_constraint"}:
                new_guide = AutoGuideList(
                    model.module._model,
                    create_plates=model.module._model.create_plates,
                )
                new_guide.append(
                    AutoNormal(
                        poutine.block(
                            model.module._model,
                            expose=[
                                "cell_time",
                                "u_read_depth",
                                "s_read_depth",
                            ],
                        ),
                        init_scale=0.1,
                    )
                )
                new_guide.append(
                    poutine.block(model.module._guide[-1], hide_types=["param"])
                )
                losses_test = model.train_faster_with_batch(
                    max_epochs=max_epochs,
                    batch_size=test_batch_size,
                    indices=test_ind,
                    new_valid_guide=new_guide,
                    log_every=log_every,
                    lr=lr,
                    use_gpu=use_gpu,
                    seed=seed,
                    elbo_name="-ELBO validation",
                )
            elif (
                guide_type
                in {
                    "velocity_auto",
                    "velocity_auto_t0_constraint",
                }
            ):  # velocity_auto, not supported (velocity_auto_depth fails with error)
                print("valid new guide")

                losses_test = model.train_faster_with_batch(
                    max_epochs=max_epochs,
                    batch_size=test_batch_size,
                    indices=test_ind,
                    log_every=log_every,
                    lr=lr,
                    use_gpu=use_gpu,
                    seed=seed,
                )
            else:
                raise
            pos_test = model.generate_posterior_samples(
                model.adata, num_samples=30, indices=test_ind, batch_size=512
            )
            fig, ax = plt.subplots()
            fig.set_size_inches(2.5, 1.5)
            ax.scatter(
                np.arange(len(losses)),
                -np.array(losses),
                label="train",
                alpha=0.25,
            )
            ax.scatter(
                np.arange(len(losses_test)),
                -np.array(losses_test),
                label="validation",
                alpha=0.25,
            )
            set_loss_plot_axes(ax)
        plt.legend()
        plt.savefig(loss_plot_path, facecolor="white", bbox_inches="tight")
        return posterior_samples, pos_test, train_ind, test_ind


def set_loss_plot_axes(ax):
    # ax.set_yscale('log')
    ax.set_yscale("symlog")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("-ELBO")
