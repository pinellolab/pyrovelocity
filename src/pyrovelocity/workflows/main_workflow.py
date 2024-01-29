from dataclasses import asdict
from dataclasses import dataclass
from datetime import timedelta

from flytekit import Resources
from flytekit import task
from flytekit import workflow
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.data import download_dataset
from pyrovelocity.interfaces import DownloadDatasetInterface
from pyrovelocity.interfaces import PreprocessDataInterface
from pyrovelocity.interfaces import PyroVelocityTrainInterface
from pyrovelocity.logging import configure_logging
from pyrovelocity.preprocess import preprocess_dataset
from pyrovelocity.train import PyroVelocityTrainInterface
from pyrovelocity.train import train_dataset


logger = configure_logging(__name__)

cache_version = "0.2.0b8"


@dataclass
class TrainingOutputs(DataClassJSONMixin):
    trained_data_path: FlyteFile
    model_path: FlyteDirectory
    posterior_samples_path: FlyteFile
    metrics_path: FlyteFile
    run_info_path: FlyteFile
    loss_plot_path: FlyteFile


@task(
    cache=False,
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
    cache=False,
    cache_version=cache_version,
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="16Gi"),
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
    timeout=timedelta(minutes=60),
    container_image="{{.image.gpu.fqn}}:{{.image.gpu.version}}",
    requests=Resources(cpu="8", mem="30Gi", ephemeral_storage="16Gi", gpu="1"),
    accelerator=T4,
)
def train_model(
    data: FlyteFile,
    data_set_name: str,
    model_identifier: str,
    train_model_args: PyroVelocityTrainInterface,
) -> TrainingOutputs:
    """
    Train model.
    """
    data_path = data.download()
    print(f"Flyte train model input data path: {data_path}")
    train_model_args.adata = str(data_path)
    (
        trained_data_path,
        model_path,
        posterior_samples_path,
        metrics_path,
        run_info_path,
        loss_plot_path,
    ) = train_dataset(
        data_set_name=data_set_name,
        model_identifier=model_identifier,
        pyrovelocity_train_model_args=train_model_args,
    )
    return TrainingOutputs(
        trained_data_path=FlyteFile(path=trained_data_path),
        model_path=FlyteDirectory(path=model_path),
        posterior_samples_path=FlyteFile(path=posterior_samples_path),
        metrics_path=FlyteFile(path=metrics_path),
        run_info_path=FlyteFile(path=run_info_path),
        loss_plot_path=FlyteFile(path=loss_plot_path),
    )


@workflow
def module_workflow(
    download_dataset_args: DownloadDatasetInterface = DownloadDatasetInterface(),
    preprocess_data_args: PreprocessDataInterface = PreprocessDataInterface(),
    train_data_set_name: str = "simulated",
    model_identifier: str = "model2",
    train_model_args: PyroVelocityTrainInterface = PyroVelocityTrainInterface(),
) -> TrainingOutputs:
    """
    Put all of the steps together into a single workflow.
    """
    data = download_data(download_dataset_args=download_dataset_args)
    processed_data = preprocess_data(
        data=data, preprocess_data_args=preprocess_data_args
    )
    model_outputs = train_model(
        data=processed_data,
        data_set_name=train_data_set_name,
        model_identifier=model_identifier,
        train_model_args=train_model_args,
    )
    return model_outputs


simulated_dataset_args = DownloadDatasetInterface(
    data_set_name="simulated",
    source="simulate",
    n_obs=60,
    n_vars=100,
)
simulated_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    adata=f"{simulated_dataset_args.data_external_path}/{simulated_dataset_args.data_set_name}.h5ad",
)
simulated_train_model1_args = PyroVelocityTrainInterface(
    guide_type="auto_t0_constraint",
    offset=False,
    cell_state="leiden",
    max_epochs=800,
)
simulated_train_model2_args = PyroVelocityTrainInterface(
    cell_state="leiden",
    max_epochs=800,
)

pancreas_dataset_args = DownloadDatasetInterface(
    data_set_name="pancreas",
    n_obs=200,
    n_vars=500,
)
pancreas_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pancreas_dataset_args.data_set_name}",
    adata=f"{pancreas_dataset_args.data_external_path}/{pancreas_dataset_args.data_set_name}.h5ad",
    process_cytotrace=True,
)
pancreas_train_model1_args = PyroVelocityTrainInterface(
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=800,
)
pancreas_train_model2_args = PyroVelocityTrainInterface(
    max_epochs=800,
)


@workflow
def training_workflow(
    simulated_dataset_args: DownloadDatasetInterface = simulated_dataset_args,
    simulated_preprocess_data_args: PreprocessDataInterface = simulated_preprocess_data_args,
    simulated_train_model1_args: PyroVelocityTrainInterface = simulated_train_model1_args,
    simulated_train_model2_args: PyroVelocityTrainInterface = simulated_train_model2_args,
    pancreas_dataset_args: DownloadDatasetInterface = pancreas_dataset_args,
    pancreas_preprocess_data_args: PreprocessDataInterface = pancreas_preprocess_data_args,
    pancreas_train_model1_args: PyroVelocityTrainInterface = pancreas_train_model1_args,
    pancreas_train_model2_args: PyroVelocityTrainInterface = pancreas_train_model2_args,
) -> list[TrainingOutputs]:
    """
    Apply the module_workflow to all datasets.
    """
    simulated_model1 = module_workflow(
        download_dataset_args=simulated_dataset_args,
        preprocess_data_args=simulated_preprocess_data_args,
        train_data_set_name="simulated",
        model_identifier="model1",
        train_model_args=simulated_train_model1_args,
    )

    simulated_model2 = module_workflow(
        download_dataset_args=simulated_dataset_args,
        preprocess_data_args=simulated_preprocess_data_args,
        train_data_set_name="simulated",
        model_identifier="model2",
        train_model_args=simulated_train_model2_args,
    )

    pancreas_model1 = module_workflow(
        download_dataset_args=pancreas_dataset_args,
        preprocess_data_args=pancreas_preprocess_data_args,
        train_data_set_name="pancreas",
        model_identifier="model1",
        train_model_args=pancreas_train_model1_args,
    )

    pancreas_model2 = module_workflow(
        download_dataset_args=pancreas_dataset_args,
        preprocess_data_args=pancreas_preprocess_data_args,
        train_data_set_name="pancreas",
        model_identifier="model2",
        train_model_args=pancreas_train_model2_args,
    )

    # pancreas_data = module_workflow(
    #     download_dataset_args=DownloadDatasetInterface(
    #         data_set_name="pancreas",
    #     ),
    #     train_dataset_name="pancreas",
    # )

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

    return [
        simulated_model1,
        simulated_model2,
        pancreas_model1,
        pancreas_model2,
    ]


if __name__ == "__main__":
    print(f"Running module_workflow() { module_workflow() }")
    print(f"Running training_workflow() { training_workflow() }")
