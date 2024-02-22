from dataclasses import asdict
from datetime import timedelta
from typing import NamedTuple
from typing import Tuple

from flytekit import Resources
from flytekit import dynamic
from flytekit import task
from flytekit import workflow
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from pyrovelocity.data import download_dataset
from pyrovelocity.interfaces import DownloadDatasetInterface
from pyrovelocity.interfaces import PreprocessDataInterface
from pyrovelocity.interfaces import PyroVelocityTrainInterface
from pyrovelocity.logging import configure_logging
from pyrovelocity.postprocess import postprocess_dataset
from pyrovelocity.preprocess import preprocess_dataset
from pyrovelocity.train import train_dataset
from pyrovelocity.workflows.main_configuration import PostprocessConfiguration
from pyrovelocity.workflows.main_configuration import ResourcesJSON
from pyrovelocity.workflows.main_configuration import TrainingOutputs
from pyrovelocity.workflows.main_configuration import WorkflowConfiguration
from pyrovelocity.workflows.main_configuration import default_resources
from pyrovelocity.workflows.main_configuration import (
    default_training_resource_limits,
)
from pyrovelocity.workflows.main_configuration import (
    default_training_resource_requests,
)
from pyrovelocity.workflows.main_configuration import default_training_resources
from pyrovelocity.workflows.main_configuration import simulated_configuration


__all__ = [
    "download_data",
    "preprocess_data",
    "train_model",
    "postprocess_data",
]

logger = configure_logging(__name__)

cache_version = "0.2.0b9"

# from pyrovelocity.workflows.main_configuration import larry_configuration
# from pyrovelocity.workflows.main_configuration import pancreas_configuration
# from pyrovelocity.workflows.main_configuration import pbmc68k_configuration
# from pyrovelocity.workflows.main_configuration import pons_configuration


@task(
    cache=False,
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
    cache=False,
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
    processed_data: FlyteFile,
    train_model_configuration: PyroVelocityTrainInterface,
) -> TrainingOutputs:
    """
    Train model.
    """
    processed_data_path = processed_data.download()
    print(f"Flyte train model input data path: {processed_data_path}")
    train_model_configuration.adata = str(processed_data_path)
    (
        data_model,
        data_model_path,
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
        data_model=data_model,
        data_model_path=data_model_path,
        trained_data_path=FlyteFile(path=trained_data_path),
        model_path=FlyteDirectory(path=model_path),
        posterior_samples_path=FlyteFile(path=posterior_samples_path),
        metrics_path=FlyteFile(path=metrics_path),
        run_info_path=FlyteFile(path=run_info_path),
        loss_plot_path=FlyteFile(path=loss_plot_path),
    )


@task(
    cache=False,
    cache_version=cache_version,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=120),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi"),
)
def postprocess_data(
    preprocess_data_args: PreprocessDataInterface,
    processed_data: FlyteFile,
    training_outputs: TrainingOutputs,
    postprocess_configuration: PostprocessConfiguration,
) -> FlyteFile:
    processed_data_path = processed_data.download()

    trained_data_path = training_outputs.trained_data_path.download()
    model_path = training_outputs.model_path.download()
    posterior_samples_path = training_outputs.posterior_samples_path.download()
    metrics_path = training_outputs.metrics_path.download()

    postprocessed_data_path = postprocess_dataset(
        data_model=training_outputs.data_model,
        data_model_path=training_outputs.data_model_path,
        processed_data_path=processed_data_path,
        trained_data_path=trained_data_path,
        model_path=model_path,
        posterior_samples_path=posterior_samples_path,
        metrics_path=metrics_path,
        vector_field_basis=preprocess_data_args.vector_field_basis,
        number_posterior_samples=postprocess_configuration.number_posterior_samples,
    )

    return FlyteFile(path=postprocessed_data_path)


@dynamic
def module_workflow(
    download_dataset_args: DownloadDatasetInterface = DownloadDatasetInterface(),
    preprocess_data_args: PreprocessDataInterface = PreprocessDataInterface(),
    train_model_configurations: list[PyroVelocityTrainInterface] = [
        PyroVelocityTrainInterface()
    ],
    postprocess_configuration: PostprocessConfiguration = PostprocessConfiguration(),
    train_model_resource_requests: ResourcesJSON = default_training_resource_requests,
    train_model_resource_limits: ResourcesJSON = default_training_resource_limits,
    postprocessing_resource_requests: ResourcesJSON = default_resources,
    postprocessing_resource_limits: ResourcesJSON = default_resources,
) -> list[FlyteFile]:
    """
    The module workflow is applied to a single dataset together with a list of
    model configurations.

    There are three ways to execute the train_model() task, which impacts how
    subsequent tasks are executed. This can also be executed with the @workflow
    decorator for a single model configuration

    ```python
    train_model_configuration: PyroVelocityTrainInterface = PyroVelocityTrainInterface(),
    ) -> TrainingOutputs:
    ...
    model_outputs = train_model(
        data=processed_data,
        train_model_configuration=train_model_configuration,
    )
    ```

    or for multiple model configurations using a map task

    ```python
    import functools
    from flytekit.experimental import map_task

    partial_train_model = functools.partial(train_model, data=processed_data)
    model_outputs = map_task(partial_train_model)(
        train_model_args=train_model_configurations,
    )
    ```

    The dynamic workflow is preferred to support both mapping of tasks over
    multiple model configurations and overriding resources from configuration
    data.
    """
    data = download_data(download_dataset_args=download_dataset_args)
    processed_data = preprocess_data(
        data=data, preprocess_data_args=preprocess_data_args
    )

    model_outputs: list[TrainingOutputs] = list()
    postprocessed_data: list[FlyteFile] = list()
    for train_model_configuration in train_model_configurations:
        model_output = train_model(
            processed_data=processed_data,
            train_model_configuration=train_model_configuration,
        ).with_overrides(
            requests=Resources(**asdict(train_model_resource_requests)),
            limits=Resources(**asdict(train_model_resource_limits)),
        )
        print(model_output)
        model_outputs.append(model_output)

        postprocessed_dataset = postprocess_data(
            preprocess_data_args=preprocess_data_args,
            processed_data=processed_data,
            training_outputs=model_output,
            postprocess_configuration=postprocess_configuration,
        ).with_overrides(
            requests=Resources(**asdict(postprocessing_resource_requests)),
            limits=Resources(**asdict(postprocessing_resource_limits)),
        )
        postprocessed_data.append(postprocessed_dataset)

    return postprocessed_data


@workflow
def training_workflow(
    simulated_configuration: WorkflowConfiguration = simulated_configuration,
    # pancreas_dataset_configuration: DownloadDatasetInterface = pancreas_configuration.download_dataset,
    # pancreas_preprocess_configuration: PreprocessDataInterface = pancreas_configuration.preprocess_data,
    # pancreas_train_model_configurations: list[
    #     PyroVelocityTrainInterface
    # ] = pancreas_configuration.training_configurations,
    # pancreas_train_model_resources: list[
    #     ResourcesJSON
    # ] = pancreas_configuration.training_resources,
    # pbmc68k_dataset_configuration: DownloadDatasetInterface = pbmc68k_configuration.download_dataset,
    # pbmc68k_preprocess_configuration: PreprocessDataInterface = pbmc68k_configuration.preprocess_data,
    # pbmc68k_train_model_configurations: list[
    #     PyroVelocityTrainInterface
    # ] = pbmc68k_configuration.training_configurations,
    # pbmc68k_train_model_resources: list[
    #     ResourcesJSON
    # ] = pbmc68k_configuration.training_resources,
    # pons_dataset_configuration: DownloadDatasetInterface = pons_configuration.download_dataset,
    # pons_preprocess_configuration: PreprocessDataInterface = pons_configuration.preprocess_data,
    # pons_train_model_configurations: list[
    #     PyroVelocityTrainInterface
    # ] = pons_configuration.training_configurations,
    # pons_train_model_resources: list[
    #     ResourcesJSON
    # ] = pons_configuration.training_resources,
    # larry_dataset_configuration: DownloadDatasetInterface = larry_configuration.download_dataset,
    # larry_preprocess_configuration: PreprocessDataInterface = larry_configuration.preprocess_data,
    # larry_train_model_configurations: list[
    #     PyroVelocityTrainInterface
    # ] = larry_configuration.training_configurations,
    # larry_train_model_resources: list[
    #     ResourcesJSON
    # ] = larry_configuration.training_resources,
) -> list[list[FlyteFile]]:
    """
    Apply the module_workflow to a collection of datasets.

    TODO: Update interface extraction to support nested dataclasses, which will
    allow simplification of input arguments to:

    simulated_configuration: WorkflowConfiguration = simulated_configuration,
    """
    simulated = module_workflow(
        download_dataset_args=simulated_configuration.download_dataset,
        preprocess_data_args=simulated_configuration.preprocess_data,
        train_model_configurations=[
            simulated_configuration.training_configuration_1,
            simulated_configuration.training_configuration_2,
        ],
        postprocess_configuration=simulated_configuration.postprocess_configuration,
        train_model_resource_requests=simulated_configuration.training_resources_requests,
        train_model_resource_limits=simulated_configuration.training_resources_limits,
        postprocessing_resource_requests=simulated_configuration.postprocessing_resources_requests,
        postprocessing_resource_limits=simulated_configuration.postprocessing_resources_limits,
    )
    # simulated = module_workflow(
    #     download_dataset_args=simulated_dataset_configuration,
    #     preprocess_data_args=simulated_preprocess_configuration,
    #     train_model_configurations=simulated_train_model_configurations,
    #     train_model_resources=simulated_train_model_resources,
    #     postprocess_model_resources=simulated_postprocess_model_resources,
    #     number_posterior_samples=simulated_number_posterior_samples,
    # )

    return [simulated]

    # pancreas = module_workflow(
    #     download_dataset_args=pancreas_dataset_configuration,
    #     preprocess_data_args=pancreas_preprocess_configuration,
    #     train_model_configurations=pancreas_train_model_configurations,
    #     train_model_resources=pancreas_train_model_resources,
    # )

    # pbmc68k = module_workflow(
    #     download_dataset_args=pbmc68k_dataset_configuration,
    #     preprocess_data_args=pbmc68k_preprocess_configuration,
    #     train_model_configurations=pbmc68k_train_model_configurations,
    #     train_model_resources=pbmc68k_train_model_resources,
    # )

    # pons = module_workflow(
    #     download_dataset_args=pons_dataset_configuration,
    #     preprocess_data_args=pons_preprocess_configuration,
    #     train_model_configurations=pons_train_model_configurations,
    #     train_model_resources=pons_train_model_resources,
    # )

    # larry = module_workflow(
    #     download_dataset_args=larry_dataset_configuration,
    #     preprocess_data_args=larry_preprocess_configuration,
    #     train_model_configurations=larry_train_model_configurations,
    #     train_model_resources=larry_train_model_resources,
    # )

    # return [
    #     simulated,
    #     pancreas,
    #     pbmc68k,
    #     pons,
    #     larry,
    # ]


if __name__ == "__main__":
    print(f"Running module_workflow() { module_workflow() }")
    print(f"Running training_workflow() { training_workflow() }")
