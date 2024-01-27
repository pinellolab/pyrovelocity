from dataclasses import asdict, make_dataclass
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple, Type

from flytekit import Resources, task, workflow
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.data import download_dataset
from pyrovelocity.logging import configure_logging
from pyrovelocity.preprocess import preprocess_dataset
from pyrovelocity.train import PyroVelocityTrainInterface, train_dataset
from pyrovelocity.workflows.configuration import create_dataclass_from_callable

logger = configure_logging(__name__)

# These can be used to override the default values of the dataclass
# but are generally only required if the callable interface lacks
# complete and dataclass-compatible type annotations.
custom_types_defaults: Dict[str, Tuple[Type, Any]] = {
    # "data_set_name": (str, "pancreas"),
    # "source": (str, "pyrovelocity"),
}

download_dataset_fields = create_dataclass_from_callable(
    download_dataset, custom_types_defaults
)

DownloadDatasetInterface = make_dataclass(
    "DownloadDatasetInterface",
    download_dataset_fields,
    bases=(DataClassJSONMixin,),
)
DownloadDatasetInterface.__module__ = __name__

cache_version = "0.2.0b8"


@task(
    cache=True,
    cache_version=cache_version,
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="2", mem="4Gi", ephemeral_storage="16Gi"),
)
def download_data(download_dataset_args: DownloadDatasetInterface) -> FlyteFile:
    """
    Download external data.
    """
    dataset_path = download_dataset(**asdict(download_dataset_args))
    return FlyteFile(path=dataset_path)


@task(
    cache=True,
    cache_version=cache_version,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="16Gi"),
)
def preprocess_data(data: FlyteFile) -> FlyteFile:
    """
    Download external data.
    """
    data_path = data.download()
    processed_dataset_path = preprocess_dataset(data_path=data_path)
    return FlyteFile(path=processed_dataset_path)


class TrainingOutputs(NamedTuple):
    trained_data_path: FlyteFile
    model_path: FlyteDirectory
    posterior_samples_path: FlyteFile
    pyrovelocity_data_path: FlyteFile
    metrics_path: FlyteFile
    run_info_path: FlyteFile


@task(
    cache=True,
    cache_version=cache_version,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=60),
    container_image="{{.image.gpu.fqn}}:{{.image.gpu.version}}",
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="16Gi", gpu="1"),
    accelerator=T4,
)
def train_model(data_set_name: str, data: FlyteFile) -> TrainingOutputs:
    """
    Train model.
    """
    data_path = data.download()
    (
        trained_data_path,
        model_path,
        posterior_samples_path,
        pyrovelocity_data_path,
        metrics_path,
        run_info_path,
    ) = train_dataset(
        data_set_name=data_set_name,
        pyrovelocity_train_model_args=PyroVelocityTrainInterface(
            adata=str(data_path),
            use_gpu=True,
        ),
    )
    return TrainingOutputs(
        trained_data_path=FlyteFile(path=trained_data_path),
        model_path=FlyteDirectory(path=model_path),
        posterior_samples_path=FlyteFile(path=posterior_samples_path),
        pyrovelocity_data_path=FlyteFile(path=pyrovelocity_data_path),
        metrics_path=FlyteFile(path=metrics_path),
        run_info_path=FlyteFile(path=run_info_path),
    )


@workflow
def module_workflow(
    download_dataset_args: DownloadDatasetInterface = DownloadDatasetInterface(),
    train_data_set_name: str = "simulated",
) -> FlyteFile:
    """
    Put all of the steps together into a single workflow.
    """
    data = download_data(download_dataset_args=download_dataset_args)
    processed_data = preprocess_data(data=data)
    model_outputs = train_model(
        data_set_name=train_data_set_name, data=processed_data
    )
    return model_outputs


@workflow
def training_workflow() -> Tuple[FlyteFile, FlyteFile, FlyteFile, FlyteFile]:
    """
    Apply the module_workflow to all datasets.
    """
    simulated_data = module_workflow(
        download_dataset_args=DownloadDatasetInterface(
            data_set_name="simulated",
            source="simulate",
        ),
        train_data_set_name="simulated",
    )

    pancreas_data = module_workflow(
        download_dataset_args=DownloadDatasetInterface(
            data_set_name="pancreas",
        ),
        train_dataset_name="pancreas",
    )

    # pons_data = module_workflow(
    #     download_dataset_args=DownloadDatasetInterface(
    #         data_set_name="pons",
    #     ),
    #     train_dataset_name="pons",
    # )

    # pbmc68k_data = module_workflow(
    #     download_dataset_args=DownloadDatasetInterface(
    #         data_set_name="pbmc68k",
    #     ),
    #     train_dataset_name="pbmc68k",
    # )

    return (
        simulated_data,
        pancreas_data,
        pons_data,
        pbmc68k_data,
    )


if __name__ == "__main__":
    print(f"Running module_workflow() { module_workflow() }")
    print(f"Running training_workflow() { training_workflow() }")
