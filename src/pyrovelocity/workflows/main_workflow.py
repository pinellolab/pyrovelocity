import os
from dataclasses import asdict
from datetime import timedelta

from flytekit import Resources, current_context, dynamic, task, workflow
from flytekit.extras.accelerators import T4, GPUAccelerator
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from returns.result import Failure, Success

from pyrovelocity.interfaces import (
    DownloadDatasetInterface,
    PreprocessDataInterface,
    PyroVelocityTrainInterface,
)
from pyrovelocity.io.archive import create_tarball_from_filtered_dir
from pyrovelocity.io.gcs import upload_file_concurrently
from pyrovelocity.logging import configure_logging
from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.summarize import summarize_dataset
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.utils import str_to_bool
from pyrovelocity.workflows.main_configuration import (
    PYROVELOCITY_SIMULATED_ONLY,
    PostprocessConfiguration,
    PostprocessOutputs,
    ResourcesJSON,
    SummarizeOutputs,
    TrainingOutputs,
    WorkflowConfiguration,
    default_resource_limits,
    default_resource_requests,
    default_training_resource_limits,
    default_training_resource_requests,
    larry_configuration,
    pancreas_configuration,
    pbmc68k_configuration,
    pons_configuration,
    simulated_configuration,
)

__all__ = [
    "download_data",
    "preprocess_data",
    "train_model",
    "postprocess_data",
    "summarize_data",
    "upload_summary",
    "module_workflow",
    "training_workflow",
]

logger = configure_logging(__name__)

CACHE_VERSION = "2024.6.27"
SUMMARIZE_CACHE_VERSION = "2024.6.27"
PYROVELOCITY_CACHE_FLAG = str_to_bool(
    os.getenv("PYROVELOCITY_CACHE_FLAG", "True")
)
L4 = GPUAccelerator("nvidia-l4")
ACCELERATOR_TYPE: GPUAccelerator = L4


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=CACHE_VERSION,
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
    print(f"\nFlyte download data path: {dataset_path}\n")
    return FlyteFile(path=dataset_path)


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=CACHE_VERSION,
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
    print(f"\nFlyte preprocess input data path: {data_path}\n")
    preprocess_data_args.adata = str(data_path)
    _, processed_dataset_path = preprocess_dataset(
        **asdict(preprocess_data_args),
    )

    print(f"\nFlyte preprocess output data path: {processed_dataset_path}\n")
    return FlyteFile(path=processed_dataset_path)


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=120),
    container_image="{{.image.gpu.fqn}}:{{.image.gpu.version}}",
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi", gpu="1"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi", gpu="1"),
    accelerator=ACCELERATOR_TYPE,
)
def train_model(
    processed_data: FlyteFile,
    train_model_configuration: PyroVelocityTrainInterface,
) -> TrainingOutputs:
    """
    Train model.
    """
    processed_data_path = processed_data.download()
    print(f"\nFlyte train model input data path: {processed_data_path}\n")
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

    print(
        f"\nFlyte train model outputs:\n"
        f"\tdata model: {data_model}\n"
        f"\tdata model path: {data_model_path}\n"
        f"\ttrained data path: {trained_data_path}\n"
        f"\tmodel path: {model_path}\n"
        f"\tposterior samples path: {posterior_samples_path}\n"
        f"\tmetrics path: {metrics_path}\n"
        f"\trun info path: {run_info_path}\n"
        f"\tloss plot path: {loss_plot_path}\n\n"
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
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=120),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi"),
)
def postprocess_data(
    preprocess_data_args: PreprocessDataInterface,
    training_outputs: TrainingOutputs,
    postprocess_configuration: PostprocessConfiguration,
) -> PostprocessOutputs:
    trained_data_path = training_outputs.trained_data_path.download()
    model_path = training_outputs.model_path.download()
    posterior_samples_path = training_outputs.posterior_samples_path.download()
    metrics_path = training_outputs.metrics_path.download()

    pyrovelocity_data_path, postprocessed_data_path = postprocess_dataset(
        data_model=training_outputs.data_model,
        data_model_path=training_outputs.data_model_path,
        trained_data_path=trained_data_path,
        model_path=model_path,
        posterior_samples_path=posterior_samples_path,
        metrics_path=metrics_path,
        vector_field_basis=preprocess_data_args.vector_field_basis,
        number_posterior_samples=postprocess_configuration.number_posterior_samples,
    )

    print(
        f"\npostprocessed_data_path: {postprocessed_data_path}\n\n",
    )

    return PostprocessOutputs(
        pyrovelocity_data=FlyteFile(path=pyrovelocity_data_path),
        postprocessed_data=FlyteFile(path=postprocessed_data_path),
    )


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=SUMMARIZE_CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=120),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi"),
)
def summarize_data(
    preprocess_data_args: PreprocessDataInterface,
    postprocessing_outputs: PostprocessOutputs,
    training_outputs: TrainingOutputs,
) -> SummarizeOutputs:
    model_path = training_outputs.model_path.download()
    pyrovelocity_data_path = postprocessing_outputs.pyrovelocity_data.download()
    postprocessed_data_path = (
        postprocessing_outputs.postprocessed_data.download()
    )

    print(
        f"\nmodel_path: {model_path}\n\n",
        f"pyrovelocity_data_path: {pyrovelocity_data_path}\n",
        f"postprocessed_data_path: {postprocessed_data_path}\n",
    )

    data_model_reports_path, dataframe_path = summarize_dataset(
        data_model=training_outputs.data_model,
        data_model_path=training_outputs.data_model_path,
        model_path=model_path,
        pyrovelocity_data_path=pyrovelocity_data_path,
        postprocessed_data_path=postprocessed_data_path,
        cell_state=preprocess_data_args.cell_state,
        vector_field_basis=preprocess_data_args.vector_field_basis,
    )
    print(
        f"\ndata_model_reports_path: {data_model_reports_path}\n",
        f"\ndataframe_path: {dataframe_path}\n\n",
    )
    return SummarizeOutputs(
        data_model_reports=FlyteDirectory(path=data_model_reports_path),
        dataframe=FlyteFile(path=dataframe_path),
    )


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=SUMMARIZE_CACHE_VERSION,
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="2", mem="4Gi", ephemeral_storage="16Gi"),
    limits=Resources(cpu="8", mem="16Gi", ephemeral_storage="200Gi"),
)
def upload_summary(
    summarize_outputs: SummarizeOutputs, training_outputs: TrainingOutputs
) -> FlyteFile:
    data_model_reports = summarize_outputs.data_model_reports
    reports_path = data_model_reports.download()
    data_model = training_outputs.data_model

    ctx = current_context()
    execution_id = ctx.execution_id.name
    archive_name = f"archive_{data_model}_{execution_id}.tar.gz"

    print(f"\nCreating tarball\n{archive_name}\nfrom {reports_path}...\n\n")

    create_tarball_from_filtered_dir(
        reports_path,
        archive_name,
    )

    upload_result = upload_file_concurrently(
        f"pyrovelocity/reports/{execution_id}",
        archive_name,
        archive_name,
    )
    if isinstance(upload_result, Success):
        file_url = upload_result.unwrap()
        print(f"\nUpload successful.\nFile URL: {file_url}\n\n")
        return FlyteFile(path=archive_name)
    elif isinstance(upload_result, Failure):
        error = upload_result.failure()
        print(
            f"\nUpload of {archive_name} failed with exception: {error}\n"
            f"Returning empty file.\n\n"
        )
        return FlyteFile(path="")


@dynamic
def module_workflow(
    download_dataset_args: DownloadDatasetInterface = DownloadDatasetInterface(),
    preprocess_data_args: PreprocessDataInterface = PreprocessDataInterface(),
    train_model_configuration_1: PyroVelocityTrainInterface = PyroVelocityTrainInterface(),
    train_model_configuration_2: PyroVelocityTrainInterface = PyroVelocityTrainInterface(),
    postprocess_configuration: PostprocessConfiguration = PostprocessConfiguration(),
    train_model_resource_requests: ResourcesJSON = default_training_resource_requests,
    train_model_resource_limits: ResourcesJSON = default_training_resource_limits,
    postprocessing_resource_requests: ResourcesJSON = default_resource_requests,
    postprocessing_resource_limits: ResourcesJSON = default_resource_limits,
    summarizing_resource_requests: ResourcesJSON = default_resource_requests,
    summarizing_resource_limits: ResourcesJSON = default_resource_limits,
    upload_results: bool = True,
) -> list[SummarizeOutputs]:
    """
    Apply the primary workflow to a single dataset with multiple model
    configurations.

    Args:
        download_dataset_args (DownloadDatasetInterface, optional): Configuration for
            pyrovelocity.tasks.data.download_dataset. Defaults to DownloadDatasetInterface().
        preprocess_data_args (PreprocessDataInterface, optional): Configuration for
            pyrovelocity.tasks.preprocess.preprocess_dataset. Defaults to PreprocessDataInterface().
        train_model_configuration_1 (PyroVelocityTrainInterface, optional): Configuration
            for pyrovelocity.train.train_dataset. Defaults to PyroVelocityTrainInterface().
        train_model_configuration_2 (PyroVelocityTrainInterface, optional): Configuration
            for pyrovelocity.train.train_dataset. Defaults to PyroVelocityTrainInterface().
        postprocess_configuration (PostprocessConfiguration, optional): Configuration for
            pyrovelocity.postprocess.postprocess_dataset. Defaults to PostprocessConfiguration().
        train_model_resource_requests (ResourcesJSON, optional): Configuration for
            flytekit.Resources. Defaults to default_training_resource_requests.
        train_model_resource_limits (ResourcesJSON, optional): Configuration for
            flytekit.Resources. Defaults to default_training_resource_limits.
        postprocessing_resource_requests (ResourcesJSON, optional): Configuration for
            flytekit.Resources. Defaults to default_resource_requests.
        postprocessing_resource_limits (ResourcesJSON, optional): Configuration for
            flytekit.Resources. Defaults to default_resource_limits.
        summarizing_resource_requests (ResourcesJSON, optional): Configuration for
            flytekit.Resources. Defaults to default_resource_requests.
        summarizing_resource_limits (ResourcesJSON, optional): Configuration for
            flytekit.Resources. Defaults to default_resource_limits.

    Returns:
        list[FlyteFile]: Workflow outputs as flytekit.types.file.FlyteFile objects.
    """
    data = download_data(download_dataset_args=download_dataset_args)
    processed_data = preprocess_data(
        data=data, preprocess_data_args=preprocess_data_args
    )

    train_model_configurations = [
        train_model_configuration_1,
        train_model_configuration_2,
    ]

    model_outputs: list[TrainingOutputs] = []
    dataset_summaries: list[SummarizeOutputs] = []
    for train_model_configuration in train_model_configurations:
        model_output = train_model(
            processed_data=processed_data,
            train_model_configuration=train_model_configuration,
        ).with_overrides(
            requests=Resources(**asdict(train_model_resource_requests)),
            limits=Resources(**asdict(train_model_resource_limits)),
        )
        model_outputs.append(model_output)

        postprocessing_outputs = postprocess_data(
            preprocess_data_args=preprocess_data_args,
            training_outputs=model_output,
            postprocess_configuration=postprocess_configuration,
        ).with_overrides(
            requests=Resources(**asdict(postprocessing_resource_requests)),
            limits=Resources(**asdict(postprocessing_resource_limits)),
        )

        dataset_summary = summarize_data(
            preprocess_data_args=preprocess_data_args,
            postprocessing_outputs=postprocessing_outputs,
            training_outputs=model_output,
        ).with_overrides(
            requests=Resources(**asdict(summarizing_resource_requests)),
            limits=Resources(**asdict(summarizing_resource_limits)),
        )

        if upload_results:
            upload_summary(
                summarize_outputs=dataset_summary,
                training_outputs=model_output,
            )

        dataset_summaries.append(dataset_summary)

    return dataset_summaries


@dynamic
def training_workflow(
    simulated_configuration: WorkflowConfiguration = simulated_configuration,
    pancreas_configuration: WorkflowConfiguration = pancreas_configuration,
    pbmc68k_configuration: WorkflowConfiguration = pbmc68k_configuration,
    pons_configuration: WorkflowConfiguration = pons_configuration,
    larry_configuration: WorkflowConfiguration = larry_configuration,
) -> list[list[SummarizeOutputs]]:
    """
    Apply the primary workflow to a collection of configurations.
    Conditionally executes configurations based on the value of PYROVELOCITY_SIMULATED_ONLY.
    """
    results = []
    configurations = [
        (simulated_configuration, "simulated"),
    ]

    if not PYROVELOCITY_SIMULATED_ONLY:
        configurations += [
            (pancreas_configuration, "pancreas"),
            (pbmc68k_configuration, "pbmc68k"),
            (pons_configuration, "pons"),
            (larry_configuration, "larry"),
        ]

    for config, _ in configurations:
        result = module_workflow(
            download_dataset_args=config.download_dataset,
            preprocess_data_args=config.preprocess_data,
            train_model_configuration_1=config.training_configuration_1,
            train_model_configuration_2=config.training_configuration_2,
            postprocess_configuration=config.postprocess_configuration,
            train_model_resource_requests=config.training_resources_requests,
            train_model_resource_limits=config.training_resources_limits,
            postprocessing_resource_requests=config.postprocessing_resources_requests,
            postprocessing_resource_limits=config.postprocessing_resources_limits,
            summarizing_resource_requests=config.summarizing_resources_requests,
            summarizing_resource_limits=config.summarizing_resources_limits,
            upload_results=config.upload_results,
        )
        results.append(result)

    return results


if __name__ == "__main__":
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
    print(f"Running module_workflow() { module_workflow() }")
    print(f"Running training_workflow() { training_workflow() }")
