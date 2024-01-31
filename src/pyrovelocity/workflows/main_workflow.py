import functools
from dataclasses import asdict
from datetime import timedelta
from typing import Dict

# from flytekit import map_task
from flytekit import Resources, dynamic, task, workflow
from flytekit.experimental import map_task
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from pyrovelocity.data import download_dataset
from pyrovelocity.interfaces import (
    DownloadDatasetInterface,
    PreprocessDataInterface,
    PyroVelocityTrainInterface,
)
from pyrovelocity.logging import configure_logging
from pyrovelocity.preprocess import preprocess_dataset
from pyrovelocity.train import train_dataset
from pyrovelocity.workflows.main_configuration import (
    ResourcesJSON,
    TrainingOutputs,
    default_training_resources,
    larry_configuration,
    pancreas_configuration,
    pbmc68k_configuration,
    pons_configuration,
    simulated_configuration,
)

logger = configure_logging(__name__)

cache_version = "0.2.0b8"


@task(
    cache=True,
    cache_version=cache_version,
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="2", mem="4Gi", ephemeral_storage="16Gi"),
    limits=Resources(cpu="8", mem="16Gi", ephemeral_storage="200Gi"),
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
    timeout=timedelta(minutes=60),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="32Gi"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi"),
)
def preprocess_data(
    data: FlyteFile, preprocess_data_args: PreprocessDataInterface
) -> FlyteFile:
    """
    Download external data.
    """
    data_path = data.download()
    print(f"Flyte preprocess input data path: {data_path}")
    preprocess_data_args.adata = str(data_path)
    _, processed_dataset_path = preprocess_dataset(
        **asdict(preprocess_data_args),
    )
    return FlyteFile(path=processed_dataset_path)


@task(
    cache=False,
    cache_version=cache_version,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=120),
    container_image="{{.image.gpu.fqn}}:{{.image.gpu.version}}",
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi", gpu="1"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi", gpu="1"),
    accelerator=T4,
)
def train_model(
    data: FlyteFile,
    train_model_configuration: PyroVelocityTrainInterface,
) -> TrainingOutputs:
    """
    Train model.
    """
    data_path = data.download()
    print(f"Flyte train model input data path: {data_path}")
    train_model_configuration.adata = str(data_path)
    (
        trained_data_path,
        model_path,
        posterior_samples_path,
        metrics_path,
        run_info_path,
        loss_plot_path,
    ) = train_dataset(
        **asdict(train_model_configuration),
    )
    return TrainingOutputs(
        trained_data_path=FlyteFile(path=trained_data_path),
        model_path=FlyteDirectory(path=model_path),
        posterior_samples_path=FlyteFile(path=posterior_samples_path),
        metrics_path=FlyteFile(path=metrics_path),
        run_info_path=FlyteFile(path=run_info_path),
        loss_plot_path=FlyteFile(path=loss_plot_path),
    )


@dynamic
def module_workflow(
    download_dataset_args: DownloadDatasetInterface = DownloadDatasetInterface(),
    preprocess_data_args: PreprocessDataInterface = PreprocessDataInterface(),
    train_model_configurations: list[PyroVelocityTrainInterface] = [
        PyroVelocityTrainInterface()
    ],
    train_model_resources: list[ResourcesJSON] = default_training_resources,
) -> list[TrainingOutputs]:
    """
    The module workflow is applied to a single dataset together with a list of
    models.

    There are three ways to execute the train_model() task, which impacts how
    subsequent tasks are executed. The first two use the @workflow decorator
    and the last requires the @dynamic decorator.


    1. For independent parallel execution treating each data-model pairing as
       requiring a separate workflow run, ensure the decorator is @workflow.
       Note that individual tasks are easily re-run in this case. For this
       reason, this is the preferred method during development, but since
       @dynamic is required in order to dynamically determine resource
       requirements, method three may be preferred in production. A single
       model configuration is required in this case and a single set of training
       outputs is produced.

    ```python
    train_model_configuration: PyroVelocityTrainInterface = PyroVelocityTrainInterface(),
    ) -> TrainingOutputs:
    model_outputs = train_model(
        data=processed_data,
        train_model_configuration=train_model_configuration,
    )
    ```

    2. For map task-based execution, ensure the decorator is @workflow.
       Note that individual tasks are not easily re-run in this case.

    ```python
    partial_train_model = functools.partial(train_model, data=processed_data)
    model_outputs = map_task(partial_train_model)(
        train_model_args=train_model_configurations,
    )
    ```

    3. For dynamic workflow-based execution, ensure the decorator is @dynamic.
       Note that individual tasks are not easily re-run in this case.
    """
    data = download_data(download_dataset_args=download_dataset_args)
    processed_data = preprocess_data(
        data=data, preprocess_data_args=preprocess_data_args
    )

    model_outputs: list[TrainingOutputs] = list()
    for train_model_configuration in train_model_configurations:
        model_output = train_model(
            data=processed_data,
            train_model_configuration=train_model_configuration,
        ).with_overrides(
            requests=Resources(**asdict(train_model_resources[0])),
            limits=Resources(**asdict(train_model_resources[1])),
        )
        model_outputs.append(model_output)
    return model_outputs


@workflow
def training_workflow(
    simulated_dataset_configuration: DownloadDatasetInterface = simulated_configuration.download_dataset,
    simulated_preprocess_configuration: PreprocessDataInterface = simulated_configuration.preprocess_data,
    simulated_train_model_configurations: list[
        PyroVelocityTrainInterface
    ] = simulated_configuration.training_configurations,
    simulated_train_model_resources: list[
        ResourcesJSON
    ] = simulated_configuration.training_resources,
    pancreas_dataset_configuration: DownloadDatasetInterface = pancreas_configuration.download_dataset,
    pancreas_preprocess_configuration: PreprocessDataInterface = pancreas_configuration.preprocess_data,
    pancreas_train_model_configurations: list[
        PyroVelocityTrainInterface
    ] = pancreas_configuration.training_configurations,
    pancreas_train_model_resources: list[
        ResourcesJSON
    ] = pancreas_configuration.training_resources,
    pbmc68k_dataset_configuration: DownloadDatasetInterface = pbmc68k_configuration.download_dataset,
    pbmc68k_preprocess_configuration: PreprocessDataInterface = pbmc68k_configuration.preprocess_data,
    pbmc68k_train_model_configurations: list[
        PyroVelocityTrainInterface
    ] = pbmc68k_configuration.training_configurations,
    pbmc68k_train_model_resources: list[
        ResourcesJSON
    ] = pbmc68k_configuration.training_resources,
    pons_dataset_configuration: DownloadDatasetInterface = pons_configuration.download_dataset,
    pons_preprocess_configuration: PreprocessDataInterface = pons_configuration.preprocess_data,
    pons_train_model_configurations: list[
        PyroVelocityTrainInterface
    ] = pons_configuration.training_configurations,
    pons_train_model_resources: list[
        ResourcesJSON
    ] = pons_configuration.training_resources,
    larry_dataset_configuration: DownloadDatasetInterface = larry_configuration.download_dataset,
    larry_preprocess_configuration: PreprocessDataInterface = larry_configuration.preprocess_data,
    larry_train_model_configurations: list[
        PyroVelocityTrainInterface
    ] = larry_configuration.training_configurations,
    larry_train_model_resources: list[
        ResourcesJSON
    ] = larry_configuration.training_resources,
) -> list[list[TrainingOutputs]]:
    """
    Apply the module_workflow to all datasets.

    TODO: update interface extraction to support nested dataclasses to simplify
    inputs to:
    simulated_configuration: WorkflowConfiguration = simulated_configuration,
    """
    simulated = module_workflow(
        download_dataset_args=simulated_dataset_configuration,
        preprocess_data_args=simulated_preprocess_configuration,
        train_model_configurations=simulated_train_model_configurations,
        train_model_resources=simulated_train_model_resources,
    )

    pancreas = module_workflow(
        download_dataset_args=pancreas_dataset_configuration,
        preprocess_data_args=pancreas_preprocess_configuration,
        train_model_configurations=pancreas_train_model_configurations,
        train_model_resources=pancreas_train_model_resources,
    )

    pbmc68k = module_workflow(
        download_dataset_args=pbmc68k_dataset_configuration,
        preprocess_data_args=pbmc68k_preprocess_configuration,
        train_model_configurations=pbmc68k_train_model_configurations,
        train_model_resources=pbmc68k_train_model_resources,
    )

    pons = module_workflow(
        download_dataset_args=pons_dataset_configuration,
        preprocess_data_args=pons_preprocess_configuration,
        train_model_configurations=pons_train_model_configurations,
        train_model_resources=pons_train_model_resources,
    )

    larry = module_workflow(
        download_dataset_args=larry_dataset_configuration,
        preprocess_data_args=larry_preprocess_configuration,
        train_model_configurations=larry_train_model_configurations,
        train_model_resources=larry_train_model_resources,
    )

    return [
        simulated,
        pancreas,
        pbmc68k,
        pons,
        larry,
    ]


if __name__ == "__main__":
    print(f"Running module_workflow() { module_workflow() }")
    print(f"Running training_workflow() { training_workflow() }")
