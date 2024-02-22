import json
import os
import uuid
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from anndata._core.anndata import AnnData
from beartype import beartype
from mlflow import MlflowClient
from numpy import ndarray

from pyrovelocity.data import load_anndata_from_path
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.models import PyroVelocity
from pyrovelocity.utils import print_anndata
from pyrovelocity.utils import print_attributes


logger = configure_logging(__name__)


@beartype
def train_dataset(
    adata: str | AnnData,
    data_set_name: str = "simulated",
    model_identifier: str = "model2",
    guide_type: str = "auto",
    model_type: str = "auto",
    batch_size: int = -1,
    use_gpu: int | bool = False,
    likelihood: str = "Poisson",
    num_samples: int = 30,
    log_every: int = 100,
    patient_improve: float = 1e-4,
    patient_init: int = 45,
    seed: int = 99,
    learning_rate: float = 0.01,
    max_epochs: int = 3000,
    include_prior: bool = True,
    library_size: bool = True,
    offset: bool = True,
    input_type: str = "raw",
    cell_specific_kinetics: Optional[str] = None,
    kinetics_num: int = 2,
    force: bool = False,
) -> Tuple[str, str, Path, Path, Path, Path, Path, Path]:
    """
    Loads processed data, trains model, and saves model and posterior samples.

    Inputs:
        "data/processed/{data_set_name}_processed.h5ad"

    Outputs:
        data:
            "models/{data_model}/trained.h5ad"
            "models/{data_model}/pyrovelocity.pkl"
            "models/{data_model}/posterior_samples.pkl.zst"
        models:
            models/{data_model}/model/
            ├── attr.pkl
            ├── model_params.pt
            ├── param_store_test.pt
            └── var_names.csv

    Args:
        data_set_name (str, optional): Data set name. Defaults to "simulated".
        model_identifier (str, optional): Model identifier. Defaults to "model1".
        force (bool, optional): Overwrite existing output. Defaults to False.

    Returns:
        Tuple[str, str, Path, Path, Path, Path, Path, Path]: Paths to saved data.

    Examples:
        >>> train_dataset() # xdoctest: +SKIP
    """
    ###########
    # load data
    ###########

    data_model = f"{data_set_name}_{model_identifier}"
    data_model_path = Path(f"models/{data_model}")

    trained_data_path = data_model_path / "trained.h5ad"
    model_path = data_model_path / "model"
    posterior_samples_path = data_model_path / "posterior_samples.pkl.zst"
    metrics_path = data_model_path / "metrics.json"
    run_info_path = data_model_path / "run_info.json"
    loss_plot_path = data_model_path / "ELBO.png"

    logger.info(f"\n\nTraining: {data_model}\n\n")

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  model data: {data_model_path}\n"
    )
    Path(data_model_path).mkdir(parents=True, exist_ok=True)

    #############
    # train model
    #############

    if torch.cuda.is_available():
        accelerators = list(range(torch.cuda.device_count()))
        use_gpu = accelerators.pop()
    else:
        use_gpu = False

    logger.info(f"GPU ID: {use_gpu}")

    if (
        os.path.isfile(trained_data_path)
        and os.path.exists(model_path)
        and os.path.isfile(posterior_samples_path)
        and not force
    ):
        logger.info(
            f"\n{trained_data_path}\n"
            f"{model_path}\n"
            f"{posterior_samples_path}\n"
            "all exist, set `force=True` to overwrite."
        )
        return (
            data_model,
            str(data_model_path),
            trained_data_path,
            model_path,
            posterior_samples_path,
            metrics_path,
            run_info_path,
            loss_plot_path,
            # cell_state,
            # vector_field_basis,
        )
    else:
        logger.info(f"Training model: {data_model}")

        # UPDATE: v2.1.1 12/26/2022 autolog only supports pytorch lightning
        # mlflow.pytorch.autolog(log_every_n_epoch=200, log_models=False, silent=False)
        with mlflow.start_run(
            run_name=f"{data_model}-{uuid.uuid4().hex[:7]}"
        ) as run:
            mlflow.set_tag(
                "mlflow.runName", f"{data_model}-{run.info.run_id[:7]}"
            )
            print(f"Active run_id: {run.info.run_id}")
            mlflow.log_params(
                {
                    "data_set_name": data_set_name,
                    "model_identifier": model_identifier,
                    "guide_type": guide_type,
                    "model_type": model_type,
                    "batch_size": batch_size,
                }
            )

            adata, trained_model, posterior_samples = train_model(
                adata=adata,
                guide_type=guide_type,
                model_type=model_type,
                batch_size=batch_size,
                use_gpu=use_gpu,
                likelihood=likelihood,
                num_samples=num_samples,
                log_every=log_every,
                patient_improve=patient_improve,
                patient_init=patient_init,
                seed=seed,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                include_prior=include_prior,
                library_size=library_size,
                offset=offset,
                input_type=input_type,
                cell_specific_kinetics=cell_specific_kinetics,
                kinetics_num=kinetics_num,
                loss_plot_path=str(loss_plot_path),
            )

            logger.info("Data attributes after model training")

            run_id = run.info.run_id

        logger.info(f"Saving posterior samples: {posterior_samples_path}\n")
        CompressedPickle.save(
            posterior_samples_path,
            posterior_samples,
        )

        ##############
        # save metrics
        ##############

        r = mlflow.get_run(run_id)

        Path(metrics_path).write_text(json.dumps(r.data.metrics, indent=4))

        Path(run_info_path).write_text(
            json.dumps(r.to_dictionary()["info"], indent=4)
        )

        log_run_info(r)

        #############
        # postprocess
        #############

        if "pancreas" in data_model:
            logger.info("checking shared time")

            def check_shared_time(posterior_samples, adata):
                adata.obs["cell_time"] = (
                    posterior_samples["cell_time"].squeeze().mean(0)
                )
                adata.obs["1-Cytotrace"] = 1 - adata.obs["cytotrace"]

            check_shared_time(posterior_samples, adata)

        # print_attributes(adata)
        print_anndata(adata)

        ##################
        # save checkpoints
        ##################

        logger.info(f"Saving trained data: {trained_data_path}")
        adata.write(trained_data_path)

        logger.info(f"Saving model: {model_path}")
        trained_model.save_model(model_path, overwrite=True)

        logger.info(
            f"\nReturning paths to saved data:\n\n"
            f"{data_model_path}\n"
            f"{trained_data_path}\n"
            f"{model_path}\n"
            f"{posterior_samples_path}\n"
            f"{metrics_path}\n"
            f"{run_info_path}\n"
            f"{loss_plot_path}\n"
        )
        return (
            data_model,
            str(data_model_path),
            trained_data_path,
            model_path,
            posterior_samples_path,
            metrics_path,
            run_info_path,
            loss_plot_path,
            # cell_state,
            # vector_field_basis,
        )


@beartype
def train_model(
    adata: str | AnnData,
    guide_type: str = "auto",
    model_type: str = "auto",
    batch_size: int = -1,
    use_gpu: int | bool = False,
    likelihood: str = "Poisson",
    num_samples: int = 30,
    log_every: int = 100,
    patient_improve: float = 1e-4,
    patient_init: int = 45,
    seed: int = 99,
    learning_rate: float = 0.01,
    max_epochs: int = 3000,
    include_prior: bool = True,
    library_size: bool = True,
    offset: bool = True,
    input_type: str = "raw",
    cell_specific_kinetics: Optional[str] = None,
    kinetics_num: int = 2,
    loss_plot_path: str = "loss_plot.png",
) -> Tuple[AnnData, PyroVelocity, Dict[str, ndarray]]:
    """
    Train a PyroVelocity model to provide probabilistic estimates of RNA velocity
    for single-cell RNA sequencing data with quantified splice variants.

    Args:
        adata (str | AnnData): Path to a file that can be read to an AnnData object or an AnnData object.
        guide_type (str, optional): The type of guide function for the Pyro model. Default is "auto".
        model_type (str, optional): The type of Pyro model. Default is "auto".
        batch_size (int, optional): Batch size for training. Default is -1, which indicates using the full dataset.
        use_gpu (int, optional): Whether to use GPU for training. Default is 0, which indicates not using GPU.
        likelihood (str, optional): Likelihood function for the Pyro model. Default is "Poisson".
        num_samples (int, optional): Number of posterior samples. Default is 30.
        log_every (int, optional): Frequency of logging progress. Default is 100.
        patient_improve (float, optional): Minimum improvement in training loss for early stopping. Default is 5e-4.
        patient_init (int, optional): Number of initial training epochs before early stopping is enabled. Default is 30.
        seed (int, optional): Random seed for reproducibility. Default is 99.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
        max_epochs (int, optional): Maximum number of training epochs. Default is 3000.
        include_prior (bool, optional): Whether to include prior information in the model. Default is True.
        library_size (bool, optional): Whether to correct for library size. Default is True.
        offset (bool, optional): Whether to add an offset to the model. Default is False.
        input_type (str, optional): Type of input data. Default is "raw".
        cell_specific_kinetics (Optional[str], optional): Name of the attribute containing cell-specific kinetics information. Default is None.
        kinetics_num (int, optional): Number of kinetics parameters. Default is 2.
        loss_plot_path (str, optional): Path to save the loss plot. Default is "loss_plot.png".

        These arguments are deprecated:
        # svi_train: bool = False,
        # train_size: float = 1.0,
        # cell_state: str = "clusters",
        # svi_train (bool, optional): Whether to use Stochastic Variational Inference for training. Default is False.
        # train_size (float, optional): Proportion of data to be used for training. Default is 1.0.
        # cell_state (str, optional): Cell state attribute in the AnnData object. Default is "clusters".

    Returns:
        Tuple[PyroVelocity, Dict[str, ndarray]]: A tuple containing the trained PyroVelocity model and a dictionary of posterior samples.

    Examples:
        >>> from pyrovelocity.train import train_model
        >>> from pyrovelocity.utils import generate_sample_data
        >>> from pyrovelocity.preprocess import copy_raw_counts
        >>> tmp = getfixture("tmp_path")
        >>> loss_plot_path = str(tmp) + "/loss_plot_docs.png"
        >>> print(loss_plot_path)
        >>> adata = generate_sample_data(random_seed=99)
        >>> copy_raw_counts(adata)
        >>> _, model, posterior_samples = train_model(adata, use_gpu=False, seed=99, max_epochs=200, loss_plot_path=loss_plot_path)
    """
    if isinstance(adata, str):
        adata = load_anndata_from_path(adata)

    print_anndata(adata)
    print_attributes(adata)

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

    if batch_size == -1:
        batch_size = adata.shape[0]

    if batch_size >= adata.shape[0]:
        losses = model.train_faster(
            max_epochs=max_epochs,
            lr=learning_rate,
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
            lr=learning_rate,
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

    return adata, model, posterior_samples


def set_loss_plot_axes(ax):
    ax.set_yscale("symlog")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("-ELBO")


def log_run_info(r: mlflow.entities.run.Run) -> None:
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    logger.info(
        f"\nrun_id: {r.info.run_id}\n"
        f"artifacts: {artifacts}\n"
        f"params: {r.data.params}\n"
        f"metrics: {r.data.metrics}\n"
        f"tags: {tags}\n"
    )
