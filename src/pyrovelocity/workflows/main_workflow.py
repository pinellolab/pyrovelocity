from dataclasses import asdict, dataclass
from datetime import timedelta

from flytekit import Resources, task, workflow
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.data import download_dataset
from pyrovelocity.interfaces import (
    DownloadDatasetInterface,
    PreprocessDataInterface,
    PyroVelocityTrainInterface,
)
from pyrovelocity.logging import configure_logging
from pyrovelocity.preprocess import preprocess_dataset
from pyrovelocity.train import PyroVelocityTrainInterface, train_dataset

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
    requests=Resources(
        cpu="16", mem="120Gi", ephemeral_storage="32Gi", gpu="1"
    ),
    limits=Resources(cpu="32", mem="200Gi", ephemeral_storage="150Gi", gpu="1"),
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


UNIFORM_MAX_EPOCHS = 2000

simulated_dataset_args = DownloadDatasetInterface(
    data_set_name="simulated",
    source="simulate",
    n_obs=3000,
    n_vars=2000,
    # n_obs=200,
    # n_vars=100,
)
simulated_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    adata=f"{simulated_dataset_args.data_external_path}/{simulated_dataset_args.data_set_name}.h5ad",
)
simulated_train_model1_args = PyroVelocityTrainInterface(
    guide_type="auto_t0_constraint",
    offset=False,
    cell_state="leiden",
    max_epochs=UNIFORM_MAX_EPOCHS,
)
simulated_train_model2_args = PyroVelocityTrainInterface(
    cell_state="leiden",
    max_epochs=UNIFORM_MAX_EPOCHS,
)


pancreas_dataset_args = DownloadDatasetInterface(
    data_set_name="pancreas",
    # n_obs=200,
    # n_vars=500,
)
pancreas_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pancreas_dataset_args.data_set_name}",
    adata=f"{pancreas_dataset_args.data_external_path}/{pancreas_dataset_args.data_set_name}.h5ad",
    process_cytotrace=True,
)
pancreas_train_model1_args = PyroVelocityTrainInterface(
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=UNIFORM_MAX_EPOCHS,
)
pancreas_train_model2_args = PyroVelocityTrainInterface(
    max_epochs=UNIFORM_MAX_EPOCHS,
)


pbmc68k_dataset_args = DownloadDatasetInterface(
    data_set_name="pbmc68k",
    # n_obs=500,
)
pbmc68k_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pbmc68k_dataset_args.data_set_name}",
    adata=f"{pbmc68k_dataset_args.data_external_path}/{pbmc68k_dataset_args.data_set_name}.h5ad",
    default_velocity_mode="stochastic",
    vector_field_basis="tsne",
)
pbmc68k_train_model1_args = PyroVelocityTrainInterface(
    guide_type="auto_t0_constraint",
    offset=False,
    cell_state="celltype",
    max_epochs=UNIFORM_MAX_EPOCHS,
)
pbmc68k_train_model2_args = PyroVelocityTrainInterface(
    cell_state="celltype",
    max_epochs=UNIFORM_MAX_EPOCHS,
)


pons_dataset_args = DownloadDatasetInterface(
    data_set_name="pons",
    # n_obs=200,
    # n_vars=500,
)
pons_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pons_dataset_args.data_set_name}",
    adata=f"{pons_dataset_args.data_external_path}/{pons_dataset_args.data_set_name}.h5ad",
)
pons_train_model1_args = PyroVelocityTrainInterface(
    guide_type="auto_t0_constraint",
    offset=False,
    cell_state="celltype",
    max_epochs=UNIFORM_MAX_EPOCHS,
)
pons_train_model2_args = PyroVelocityTrainInterface(
    cell_state="celltype",
    max_epochs=UNIFORM_MAX_EPOCHS,
)


larry_dataset_args = DownloadDatasetInterface(
    data_set_name="larry",
    # n_obs=500,
    # n_vars=2000,
)
larry_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{larry_dataset_args.data_set_name}",
    adata=f"{larry_dataset_args.data_external_path}/{larry_dataset_args.data_set_name}.h5ad",
    vector_field_basis="emb",
)
larry_train_model1_args = PyroVelocityTrainInterface(
    guide_type="auto_t0_constraint",
    svi_train=True,
    batch_size=4000,
    offset=False,
    cell_state="state_info",
    max_epochs=UNIFORM_MAX_EPOCHS,
)
larry_train_model2_args = PyroVelocityTrainInterface(
    svi_train=True,
    batch_size=4000,
    cell_state="state_info",
    max_epochs=UNIFORM_MAX_EPOCHS,
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
    pbmc68k_dataset_args: DownloadDatasetInterface = pbmc68k_dataset_args,
    pbmc68k_preprocess_data_args: PreprocessDataInterface = pbmc68k_preprocess_data_args,
    pbmc68k_train_model1_args: PyroVelocityTrainInterface = pbmc68k_train_model1_args,
    pbmc68k_train_model2_args: PyroVelocityTrainInterface = pbmc68k_train_model2_args,
    pons_dataset_args: DownloadDatasetInterface = pons_dataset_args,
    pons_preprocess_data_args: PreprocessDataInterface = pons_preprocess_data_args,
    pons_train_model1_args: PyroVelocityTrainInterface = pons_train_model1_args,
    pons_train_model2_args: PyroVelocityTrainInterface = pons_train_model2_args,
    larry_dataset_args: DownloadDatasetInterface = larry_dataset_args,
    larry_preprocess_data_args: PreprocessDataInterface = larry_preprocess_data_args,
    larry_train_model1_args: PyroVelocityTrainInterface = larry_train_model1_args,
    larry_train_model2_args: PyroVelocityTrainInterface = larry_train_model2_args,
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

    pbmc68k_model1 = module_workflow(
        download_dataset_args=pbmc68k_dataset_args,
        preprocess_data_args=pbmc68k_preprocess_data_args,
        train_data_set_name="pbmc68k",
        model_identifier="model1",
        train_model_args=pbmc68k_train_model1_args,
    )

    pbmc68k_model2 = module_workflow(
        download_dataset_args=pbmc68k_dataset_args,
        preprocess_data_args=pbmc68k_preprocess_data_args,
        train_data_set_name="pbmc68k",
        model_identifier="model2",
        train_model_args=pbmc68k_train_model2_args,
    )

    pons_model1 = module_workflow(
        download_dataset_args=pons_dataset_args,
        preprocess_data_args=pons_preprocess_data_args,
        train_data_set_name="pons",
        model_identifier="model1",
        train_model_args=pons_train_model1_args,
    )

    pons_model2 = module_workflow(
        download_dataset_args=pons_dataset_args,
        preprocess_data_args=pons_preprocess_data_args,
        train_data_set_name="pons",
        model_identifier="model2",
        train_model_args=pons_train_model2_args,
    )

    larry_model1 = module_workflow(
        download_dataset_args=larry_dataset_args,
        preprocess_data_args=larry_preprocess_data_args,
        train_data_set_name="larry",
        model_identifier="model1",
        train_model_args=larry_train_model1_args,
    )

    larry_model2 = module_workflow(
        download_dataset_args=larry_dataset_args,
        preprocess_data_args=larry_preprocess_data_args,
        train_data_set_name="larry",
        model_identifier="model2",
        train_model_args=larry_train_model2_args,
    )

    return [
        simulated_model1,
        simulated_model2,
        pancreas_model1,
        pancreas_model2,
        pbmc68k_model1,
        pbmc68k_model2,
        pons_model1,
        pons_model2,
        larry_model1,
        larry_model2,
    ]


if __name__ == "__main__":
    print(f"Running module_workflow() { module_workflow() }")
    print(f"Running training_workflow() { training_workflow() }")
