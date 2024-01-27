import json
import os
import uuid
from dataclasses import asdict, make_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import mlflow
import torch
from beartype import beartype
from mashumaro.mixins.json import DataClassJSONMixin
from mlflow import MlflowClient

from pyrovelocity.api import train_model
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.utils import print_anndata, print_attributes
from pyrovelocity.workflows.configuration import create_dataclass_from_callable

logger = configure_logging(__name__)

pyrovelocity_train_types_defaults: Dict[str, Tuple[Type, Any]] = {
    "adata": (str, "/data/processed/simulated_processed.h5ad"),
}

pyrovelocity_train_fields = create_dataclass_from_callable(
    train_model,
    pyrovelocity_train_types_defaults,
)

PyroVelocityTrainInterface = make_dataclass(
    "PyroVelocityTrainInterface",
    pyrovelocity_train_fields,
    bases=(DataClassJSONMixin,),
)
PyroVelocityTrainInterface.__module__ = __name__


@beartype
def train_dataset(
    data_set_name: str = "simulated",
    model_identifier: str = "model1",
    pyrovelocity_train_model_args: Optional[PyroVelocityTrainInterface] = None,
    force: bool = False,
) -> Tuple[Path, Path, Path, Path, Path, Path]:
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
        pyrovelocity_train_model_args (Optional[PyroVelocityTrainInterface], optional): Arguments passed to train_model. Defaults to None.
        force (bool, optional): Overwrite existing output. Defaults to False.

    Returns:
        Tuple[Path, Path, Path, Path, Path, Path]: Paths to saved data.

    Examples:
        >>> train_dataset() # xdoctest: +SKIP
    """
    ###########
    # load data
    ###########

    data_model = f"{data_set_name}_{model_identifier}"
    model_dir = Path(f"models/{data_model}")

    if pyrovelocity_train_model_args is None:
        processed_path = Path(f"data/processed/{data_set_name}_processed.h5ad")
        pyrovelocity_train_model_args = PyroVelocityTrainInterface(
            adata=str(processed_path)
        )

    trained_data_path = model_dir / "trained.h5ad"
    model_path = model_dir / "model"
    posterior_samples_path = model_dir / "posterior_samples.pkl.zst"
    pyrovelocity_data_path = model_dir / "pyrovelocity.pkl.zst"
    metrics_path = model_dir / "metrics.json"
    run_info_path = model_dir / "run_info.json"

    logger.info(f"\n\nTraining: {data_model}\n\n")

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  model data: {model_dir}\n"
    )
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    #############
    # train model
    #############

    if torch.cuda.is_available():
        accelerators = list(range(torch.cuda.device_count()))
        gpu_id = accelerators.pop()
    else:
        gpu_id = False

    logger.info(f"GPU ID: {gpu_id}")
    pyrovelocity_train_model_args.use_gpu = gpu_id

    if (
        os.path.isfile(trained_data_path)
        and os.path.exists(model_path)
        and os.path.isfile(pyrovelocity_data_path)
        and os.path.isfile(posterior_samples_path)
        and not force
    ):
        logger.info(
            f"\n{trained_data_path}\n"
            f"{model_path}\n"
            f"{pyrovelocity_data_path}\n"
            f"{posterior_samples_path}\n"
            "all exist, set `force=True` to overwrite."
        )
        return (
            trained_data_path,
            model_path,
            posterior_samples_path,
            pyrovelocity_data_path,
            metrics_path,
            run_info_path,
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
            mlflow.log_params(asdict(pyrovelocity_train_model_args))

            adata, trained_model, posterior_samples = train_model(
                **asdict(pyrovelocity_train_model_args),
            )

            logger.info("Data attributes after model training")

            run_id = run.info.run_id

        logger.info(
            f"\nSaving pyrovelocity data: {pyrovelocity_data_path}\n"
            f"Saving posterior samples: {posterior_samples_path}\n"
        )
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

        print_attributes(adata)
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
            f"{trained_data_path}\n"
            f"{model_path}\n"
            f"{posterior_samples_path}\n"
            f"{pyrovelocity_data_path}\n"
            f"{metrics_path}\n"
            f"{run_info_path}\n"
        )
        return (
            trained_data_path,
            model_path,
            posterior_samples_path,
            pyrovelocity_data_path,
            metrics_path,
            run_info_path,
        )


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
