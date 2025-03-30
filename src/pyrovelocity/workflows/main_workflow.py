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
from pyrovelocity.io.metrics import (
    add_duration_to_run_info,
    combine_json_files,
    generate_and_save_metric_tables,
    load_json,
)
from pyrovelocity.logging import configure_logging
from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.summarize import summarize_dataset
from pyrovelocity.tasks.time_fate_correlation import (
    create_time_lineage_fate_correlation_plot,
)
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
    SummarizeConfiguration,
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
POSTPROCESS_CACHE_VERSION = f"{CACHE_VERSION}.2"
SUMMARIZE_CACHE_VERSION = f"{CACHE_VERSION}.3"
UPLOAD_CACHE_VERSION = f"{CACHE_VERSION}.7"
LINEAGE_FATE_CORRELATION_CACHE_VERSION = f"{CACHE_VERSION}.8"
COMBINE_METRICS_CACHE_VERSION = f"{CACHE_VERSION}.5"
DEFAULT_ACCELERATOR_TYPE: GPUAccelerator = T4


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
    accelerator=DEFAULT_ACCELERATOR_TYPE,
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
    summarize_configuration: SummarizeConfiguration,
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
        selected_genes=summarize_configuration.selected_genes,
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
        data_model=str(training_outputs.data_model),
        data_model_reports=FlyteDirectory(path=str(data_model_reports_path)),
        dataframe=FlyteFile(path=str(dataframe_path)),
        run_metrics_path=FlyteFile(path=str(metrics_path)),
        run_info_path=FlyteFile(path=str(run_info_path)),
        loss_plot_path=FlyteFile(path=str(loss_plot_path)),
        loss_csv_path=FlyteFile(path=str(loss_csv_path)),
        combined_metrics_path=FlyteFile(path=str(combined_metrics_path)),
        pyrovelocity_data=FlyteFile(path=str(pyrovelocity_data_path)),
        postprocessed_data=FlyteFile(path=str(postprocessed_data_path)),
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
            ".dill.zst",
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
    summarize_configuration: SummarizeConfiguration = SummarizeConfiguration(),
    train_model_resource_requests: ResourcesJSON = default_training_resource_requests,
    train_model_resource_limits: ResourcesJSON = default_training_resource_limits,
    postprocessing_resource_requests: ResourcesJSON = default_resource_requests,
    postprocessing_resource_limits: ResourcesJSON = default_resource_limits,
    summarizing_resource_requests: ResourcesJSON = default_resource_requests,
    summarizing_resource_limits: ResourcesJSON = default_resource_limits,
    accelerator_type: str = "nvidia-tesla-t4",
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

    model_outputs: list[SummarizeOutputs] = []
    for train_model_configuration in train_model_configurations:
        training_output = train_model(
            preprocess_outputs=processed_outputs,
            train_model_configuration=train_model_configuration,
        ).with_overrides(
            requests=Resources(**asdict(train_model_resource_requests)),
            limits=Resources(**asdict(train_model_resource_limits)),
            accelerator=GPUAccelerator(accelerator_type),
        )

        postprocessing_output = postprocess_data(
            preprocess_data_args=preprocess_data_args,
            training_outputs=training_output,
            postprocess_configuration=postprocess_configuration,
        ).with_overrides(
            requests=Resources(**asdict(postprocessing_resource_requests)),
            limits=Resources(**asdict(postprocessing_resource_limits)),
        )

        summarize_output = summarize_data(
            preprocess_data_args=preprocess_data_args,
            postprocessing_outputs=postprocessing_output,
            training_outputs=training_output,
            summarize_configuration=summarize_configuration,
        ).with_overrides(
            requests=Resources(**asdict(summarizing_resource_requests)),
            limits=Resources(**asdict(summarizing_resource_limits)),
        )

        if upload_results:
            upload_summary(
                training_outputs=training_output,
                summarize_outputs=summarize_output,
            )

        model_outputs.append(summarize_output)

    return model_outputs


@task(
    cache=PYROVELOCITY_CACHE_FLAG,
    cache_version=LINEAGE_FATE_CORRELATION_CACHE_VERSION,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=120),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="100Gi"),
    limits=Resources(cpu="16", mem="100Gi", ephemeral_storage="500Gi"),
    enable_deck=False,
)
def combine_time_lineage_fate_correlation(
    results: List[List[SummarizeOutputs]],
) -> List[FlyteFile]:
    print(results)
    model_ordered_results = list(map(list, zip(*results)))
    print(model_ordered_results)
    time_lineage_fate_correlation_plots = []

    for model_results in model_ordered_results:
        print(model_results)

        prepared_model_results = []
        for model_output in model_results:
            postprocessed_data_path = model_output.postprocessed_data.download()
            posterior_samples_path = model_output.pyrovelocity_data.download()

            prepared_model_results.append(
                {
                    "data_model": model_output.data_model,
                    "postprocessed_data": postprocessed_data_path,
                    "pyrovelocity_data": posterior_samples_path,
                }
            )

        time_lineage_fate_correlation_plot = (
            create_time_lineage_fate_correlation_plot(
                model_results=prepared_model_results,
                vertical_texts=[
                    "Monocytes",
                    "Neutrophils",
                    "Multilineage",
                    "All lineages",
                ][: len(prepared_model_results)],
                reports_path=Path("."),
            )
        )

        time_lineage_fate_correlation_plots.append(
            time_lineage_fate_correlation_plot
        )

    ctx = current_context()
    execution_id = ctx.execution_id.name

    attempted_upload_results = []

    for file in time_lineage_fate_correlation_plots:
        for ext in ["", ".png"]:
            upload_result = upload_file_concurrently(
                bucket_name=f"pyrovelocity/reports/{execution_id}",
                source_filename=f"{file}{ext}",
                destination_blob_name=f"{file}{ext}",
            )
            attempted_upload_results.append(upload_result)

    if all(isinstance(result, Success) for result in attempted_upload_results):
        logger.info("\nAll time lineage fate correlation uploads successful.")
    else:
        logger.error(
            "\nOne or more time lineage fate correlation uploads failed."
        )
        failed_uploads = [
            str(file)
            for file, result in zip(
                time_lineage_fate_correlation_plots, attempted_upload_results
            )
            if isinstance(result, Failure)
        ]
        logger.info(f"Failed uploads: {', '.join(failed_uploads)}")

    return [FlyteFile(path=str(x)) for x in time_lineage_fate_correlation_plots]


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

    files_to_upload = generate_and_save_metric_tables(
        json_data=combined_metrics,
        output_dir=Path("."),
    )

    ctx = current_context()
    execution_id = ctx.execution_id.name

    attempted_upload_results = []

    for file in files_to_upload:
        upload_result = upload_file_concurrently(
            bucket_name=f"pyrovelocity/reports/{execution_id}",
            source_filename=file,
            destination_blob_name=str(file),
        )
        attempted_upload_results.append(upload_result)

    if all(isinstance(result, Success) for result in attempted_upload_results):
        print("\nAll uploads successful.")
        return CombinedMetricsOutputs(
            metrics_json=FlyteFile(path=str(files_to_upload[0])),
            metrics_html=FlyteFile(path=str(files_to_upload[1])),
            metrics_latex=FlyteFile(path=str(files_to_upload[2])),
            metrics_md=FlyteFile(path=str(files_to_upload[3])),
            elbo_html=FlyteFile(path=str(files_to_upload[4])),
            elbo_latex=FlyteFile(path=str(files_to_upload[5])),
            elbo_md=FlyteFile(path=str(files_to_upload[6])),
            mae_html=FlyteFile(path=str(files_to_upload[7])),
            mae_latex=FlyteFile(path=str(files_to_upload[8])),
            mae_md=FlyteFile(path=str(files_to_upload[9])),
        )
    else:
        print("\nOne or more uploads failed.")
        failed_uploads = [
            str(file)
            for file, result in zip(files_to_upload, attempted_upload_results)
            if isinstance(result, Failure)
        ]
        print(f"Failed uploads: {', '.join(failed_uploads)}")
        return CombinedMetricsOutputs(
            metrics_json=FlyteFile(path=""),
            metrics_latex=FlyteFile(path=""),
            metrics_html=FlyteFile(path=""),
            metrics_md=FlyteFile(path=""),
            elbo_html=FlyteFile(path=""),
            elbo_latex=FlyteFile(path=""),
            elbo_md=FlyteFile(path=""),
            mae_html=FlyteFile(path=""),
            mae_latex=FlyteFile(path=""),
            mae_md=FlyteFile(path=""),
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

    stationary_configurations = [
        (pbmc5k_configuration, "pbmc5k"),
        (pbmc10k_configuration, "pbmc10k"),
        (pbmc68k_configuration, "pbmc68k"),
    ]

    developmental_configurations = [
        (bonemarrow_configuration, "bonemarrow"),
        (pancreas_configuration, "pancreas"),
        (pons_configuration, "pons"),
    ]

    lineage_traced_results = []
    lineage_traced_configurations = [
        (larry_mono_configuration, "larry_mono"),
        (larry_neu_configuration, "larry_neu"),
        (larry_multilineage_configuration, "larry_multilineage"),
        (larry_configuration, "larry"),
    ]

    configurations = [
        (simulated_configuration, "simulated"),
    ]

    if not PYROVELOCITY_DATA_SUBSET:
        configurations += stationary_configurations
        configurations += developmental_configurations
        configurations += lineage_traced_configurations

    for config, data_set_name in configurations:
        result = map_model_configurations_over_data_set(
            download_dataset_args=config.download_dataset,
            preprocess_data_args=config.preprocess_data,
            train_model_configuration_1=config.training_configuration_1,
            train_model_configuration_2=config.training_configuration_2,
            postprocess_configuration=config.postprocess_configuration,
            summarize_configuration=config.summarize_configuration,
            train_model_resource_requests=config.training_resources_requests,
            train_model_resource_limits=config.training_resources_limits,
            postprocessing_resource_requests=config.postprocessing_resources_requests,
            postprocessing_resource_limits=config.postprocessing_resources_limits,
            summarizing_resource_requests=config.summarizing_resources_requests,
            summarizing_resource_limits=config.summarizing_resources_limits,
            accelerator_type=config.accelerator_type,
            upload_results=config.upload_results,
        )
        if "larry" in data_set_name:
            lineage_traced_results.append(result)
        results.append(result)

    combine_all_metrics(results=results)

    if len(lineage_traced_results) > 0:
        combine_time_lineage_fate_correlation(results=lineage_traced_results)

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
