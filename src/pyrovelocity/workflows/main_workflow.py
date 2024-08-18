import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from beartype.typing import List
from flytekit import Resources, current_context, dynamic, task
from flytekit.extras.accelerators import T4, GPUAccelerator
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from returns.result import Failure, Success

from pyrovelocity.interfaces import (
    DownloadDatasetInterface,
    PreprocessDataInterface,
    PyroVelocityTrainInterface,
)
from pyrovelocity.io.archive import (
    copy_files_to_directory,
    create_tarball_from_filtered_dir,
)
from pyrovelocity.io.gcs import upload_file_concurrently
from pyrovelocity.io.json import (
    add_duration_to_run_info,
    combine_json_files,
    generate_tables,
    load_json,
)
from pyrovelocity.logging import configure_logging
from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.summarize import summarize_dataset
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.workflows.constants import (
    PYROVELOCITY_CACHE_FLAG,
    PYROVELOCITY_DATA_SUBSET,
)
from pyrovelocity.workflows.main_configuration import (
    CombinedMetricsOutputs,
    PostprocessConfiguration,
    PostprocessOutputs,
    PreprocessOutputs,
    ResourcesJSON,
    SummarizeOutputs,
    TrainingOutputs,
    WorkflowConfiguration,
    bonemarrow_configuration,
    default_resource_limits,
    default_resource_requests,
    default_training_resource_limits,
    default_training_resource_requests,
    larry_configuration,
    larry_mono_configuration,
    larry_multilineage_configuration,
    larry_neu_configuration,
    pancreas_configuration,
    pbmc5k_configuration,
    pbmc10k_configuration,
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
    "map_model_configurations_over_data_set",
    "training_workflow",
]

logger = configure_logging(__name__)

CACHE_VERSION = "2024.8.15"
DOWNLOAD_CACHE_VERSION = f"{CACHE_VERSION}.0"
PREPROCESS_CACHE_VERSION = f"{CACHE_VERSION}.0"
TRAIN_CACHE_VERSION = f"{CACHE_VERSION}.0"
POSTPROCESS_CACHE_VERSION = f"{CACHE_VERSION}.0"
SUMMARIZE_CACHE_VERSION = f"{CACHE_VERSION}.0"
UPLOAD_CACHE_VERSION = f"{CACHE_VERSION}.3"
COMBINE_METRICS_CACHE_VERSION = f"{CACHE_VERSION}.3"
L4 = GPUAccelerator("nvidia-l4")
ACCELERATOR_TYPE: GPUAccelerator = T4


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=DOWNLOAD_CACHE_VERSION,
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
    return FlyteFile(path=str(dataset_path))


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=PREPROCESS_CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=60),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="32Gi"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi"),
    enable_deck=False,
)
def preprocess_data(
    data: FlyteFile, preprocess_data_args: PreprocessDataInterface
) -> PreprocessOutputs:
    """
    Download external data.
    """
    data_path = data.download()
    print(f"\nFlyte preprocess input data path: {data_path}\n")
    preprocess_data_args.adata = str(data_path)
    _, processed_dataset_path, processed_reports_path = preprocess_dataset(
        **asdict(preprocess_data_args),
    )

    print(f"\nFlyte preprocess output data path: {processed_dataset_path}\n")
    return PreprocessOutputs(
        processed_data=FlyteFile(path=str(processed_dataset_path)),
        processed_reports=FlyteDirectory(path=str(processed_reports_path)),
    )


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=TRAIN_CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=240),
    container_image="{{.image.gpu.fqn}}:{{.image.gpu.version}}",
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi", gpu="1"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi", gpu="1"),
    accelerator=ACCELERATOR_TYPE,
    enable_deck=False,
)
def train_model(
    preprocess_outputs: PreprocessOutputs,
    train_model_configuration: PyroVelocityTrainInterface,
) -> TrainingOutputs:
    """
    Train model.
    """
    processed_data_path = preprocess_outputs.processed_data.download()
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
        loss_csv_path,
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
        f"\tloss csv path: {loss_csv_path}\n\n"
    )

    return TrainingOutputs(
        data_model=data_model,
        data_model_path=data_model_path,
        trained_data_path=FlyteFile(path=str(trained_data_path)),
        model_path=FlyteDirectory(path=str(model_path)),
        posterior_samples_path=FlyteFile(path=str(posterior_samples_path)),
        metrics_path=FlyteFile(path=str(metrics_path)),
        run_info_path=FlyteFile(path=str(run_info_path)),
        loss_plot_path=FlyteFile(path=str(loss_plot_path)),
        loss_csv_path=FlyteFile(path=str(loss_csv_path)),
    )


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=POSTPROCESS_CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=240),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi"),
    enable_deck=False,
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
        pyrovelocity_data=FlyteFile(path=str(pyrovelocity_data_path)),
        postprocessed_data=FlyteFile(path=str(postprocessed_data_path)),
        metrics_path=FlyteFile(path=str(metrics_path)),
    )


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=SUMMARIZE_CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=180),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="50Gi"),
    limits=Resources(cpu="16", mem="60Gi", ephemeral_storage="200Gi"),
    enable_deck=False,
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
    run_info_path = training_outputs.run_info_path.download()
    metrics_path = postprocessing_outputs.metrics_path.download()
    loss_plot_path = training_outputs.loss_plot_path.download()
    loss_csv_path = training_outputs.loss_csv_path.download()

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

    data_model_metrics_path = Path(data_model_reports_path) / "metrics"
    copy_files_to_reports = [
        run_info_path,
        metrics_path,
        loss_plot_path,
        loss_csv_path,
    ]

    copy_files_result = copy_files_to_directory(
        files_to_copy=copy_files_to_reports,
        target_directory=data_model_metrics_path,
    )
    if isinstance(copy_files_result, Failure):
        print(
            f"\nError copying files to {data_model_reports_path}: {copy_files_result.failure()}\n\n"
        )

    add_duration_to_run_info(run_info_path)
    combined_metrics_path = Path(data_model_reports_path) / "metrics.json"
    combine_json_result = combine_json_files(
        file1=run_info_path,
        file2=metrics_path,
        output_file=combined_metrics_path,
    )
    if isinstance(combine_json_result, Failure):
        print(
            f"\nError combining metrics and run info files in {combined_metrics_path}:\n",
            f"{combine_json_result.failure()}\n\n",
        )

    return SummarizeOutputs(
        data_model_reports=FlyteDirectory(path=str(data_model_reports_path)),
        dataframe=FlyteFile(path=str(dataframe_path)),
        run_metrics_path=FlyteFile(path=str(metrics_path)),
        run_info_path=FlyteFile(path=str(run_info_path)),
        loss_plot_path=FlyteFile(path=str(loss_plot_path)),
        loss_csv_path=FlyteFile(path=str(loss_csv_path)),
        combined_metrics_path=FlyteFile(path=str(combined_metrics_path)),
    )


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=UPLOAD_CACHE_VERSION,
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="2", mem="4Gi", ephemeral_storage="16Gi"),
    limits=Resources(cpu="8", mem="16Gi", ephemeral_storage="200Gi"),
)
def upload_summary(
    summarize_outputs: SummarizeOutputs,
    training_outputs: TrainingOutputs,
) -> FlyteFile:
    data_model_reports = summarize_outputs.data_model_reports
    reports_path = data_model_reports.download()
    data_model = training_outputs.data_model

    ctx = current_context()
    execution_id = ctx.execution_id.name
    archive_name = f"archive_{data_model}_{execution_id}.tar.gz"

    print(f"\nCreating tarball\n{archive_name}\nfrom {reports_path}...\n\n")

    create_tarball_from_filtered_dir(
        src_dir=reports_path,
        output_filename=archive_name,
        extensions=(
            ".csv",
            ".json",
            ".pdf",
            ".png",
        ),
    )

    upload_result = upload_file_concurrently(
        bucket_name=f"pyrovelocity/reports/{execution_id}",
        source_filename=archive_name,
        destination_blob_name=archive_name,
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
def map_model_configurations_over_data_set(
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
    processed_outputs = preprocess_data(
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
            preprocess_outputs=processed_outputs,
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


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=COMBINE_METRICS_CACHE_VERSION,
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="2", mem="4Gi", ephemeral_storage="8Gi"),
    limits=Resources(cpu="4", mem="8Gi", ephemeral_storage="16Gi"),
)
def combine_all_metrics(
    results: List[List[SummarizeOutputs]]
) -> CombinedMetricsOutputs:
    combined_metrics = {}

    for dataset_results in results:
        for model_result in dataset_results:
            metrics_path = model_result.combined_metrics_path.download()
            metrics_result = load_json(Path(metrics_path))

            if isinstance(metrics_result, Success):
                metrics = metrics_result.unwrap()
                run_name = metrics.get("run_name", "unknown")
                combined_metrics[run_name] = metrics
            else:
                print(
                    f"Failed to load metrics from {metrics_path}: {metrics_result.failure()}"
                )

    json_metrics_file = Path("combined_metrics.json")
    with json_metrics_file.open("w") as f:
        json.dump(combined_metrics, f, indent=2)

    latex_table, html_table, markdown_table, _ = generate_tables(
        combined_metrics
    )

    latex_metrics_file = Path("combined_metrics_table.tex")
    with latex_metrics_file.open("w") as f:
        f.write(latex_table)

    html_metrics_file = Path("combined_metrics_table.html")
    with html_metrics_file.open("w") as f:
        f.write(html_table)

    md_metrics_file = Path("combined_metrics_table.md")
    with md_metrics_file.open("w") as f:
        f.write(markdown_table)

    ctx = current_context()
    execution_id = ctx.execution_id.name

    files_to_upload = [
        json_metrics_file,
        latex_metrics_file,
        html_metrics_file,
        md_metrics_file,
    ]
    upload_results = []

    for file in files_to_upload:
        upload_result = upload_file_concurrently(
            bucket_name=f"pyrovelocity/reports/{execution_id}",
            source_filename=file,
            destination_blob_name=str(file),
        )
        upload_results.append(upload_result)

    if all(isinstance(result, Success) for result in upload_results):
        print("\nAll uploads successful.")
        return CombinedMetricsOutputs(
            json_metrics=FlyteFile(path=str(json_metrics_file)),
            latex_metrics=FlyteFile(path=str(latex_metrics_file)),
            html_metrics=FlyteFile(path=str(html_metrics_file)),
            md_metrics=FlyteFile(path=str(md_metrics_file)),
        )
    else:
        print("\nOne or more uploads failed.")
        failed_uploads = [
            str(file)
            for file, result in zip(files_to_upload, upload_results)
            if isinstance(result, Failure)
        ]
        print(f"Failed uploads: {', '.join(failed_uploads)}")
        return CombinedMetricsOutputs(
            json_metrics=FlyteFile(path=""),
            latex_metrics=FlyteFile(path=""),
            html_metrics=FlyteFile(path=""),
            md_metrics=FlyteFile(path=""),
        )


@dynamic
def training_workflow(
    simulated_configuration: WorkflowConfiguration = simulated_configuration,
    pancreas_configuration: WorkflowConfiguration = pancreas_configuration,
    bonemarrow_configuration: WorkflowConfiguration = bonemarrow_configuration,
    pbmc5k_configuration: WorkflowConfiguration = pbmc5k_configuration,
    pbmc10k_configuration: WorkflowConfiguration = pbmc10k_configuration,
    pbmc68k_configuration: WorkflowConfiguration = pbmc68k_configuration,
    pons_configuration: WorkflowConfiguration = pons_configuration,
    larry_configuration: WorkflowConfiguration = larry_configuration,
    larry_neu_configuration: WorkflowConfiguration = larry_neu_configuration,
    larry_mono_configuration: WorkflowConfiguration = larry_mono_configuration,
    larry_multilineage_configuration: WorkflowConfiguration = larry_multilineage_configuration,
) -> list[list[SummarizeOutputs]]:
    """
    Apply the primary workflow to a collection of configurations.
    Conditionally executes configurations based on the value of PYROVELOCITY_DATA_SUBSET.
    """
    results = []
    configurations = [
        (simulated_configuration, "simulated"),
    ]

    if not PYROVELOCITY_DATA_SUBSET:
        configurations += [
            (pancreas_configuration, "pancreas"),
            (bonemarrow_configuration, "bonemarrow"),
            (pbmc5k_configuration, "pbmc5k"),
            (pbmc10k_configuration, "pbmc10k"),
            (pbmc68k_configuration, "pbmc68k"),
            (pons_configuration, "pons"),
            (larry_configuration, "larry"),
            (larry_neu_configuration, "larry_neu"),
            (larry_mono_configuration, "larry_mono"),
            (larry_multilineage_configuration, "larry_multilineage"),
        ]

    for config, _ in configurations:
        result = map_model_configurations_over_data_set(
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

    combine_all_metrics(results=results)

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
    print(f"Running training_workflow() { training_workflow() }")
