import os
from dataclasses import dataclass, field

from beartype import beartype
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.interfaces import (
    DownloadDatasetInterface,
    PreprocessDataInterface,
    PyroVelocityTrainInterface,
)
from pyrovelocity.logging import configure_logging
from pyrovelocity.utils import str_to_bool

__all__ = [
    "ResourcesJSON",
    "PostprocessConfiguration",
    "TrainingOutputs",
    "PostprocessOutputs",
    "SummarizeOutputs",
    "WorkflowConfiguration",
]


logger = configure_logging(__name__)


PYROVELOCITY_TESTING_FLAG = str_to_bool(
    os.getenv("PYROVELOCITY_TESTING_FLAG", "True")
)
PYROVELOCITY_SIMULATED_ONLY = str_to_bool(
    os.getenv("PYROVELOCITY_SIMULATED_ONLY", "True")
)
PYROVELOCITY_UPLOAD_RESULTS = str_to_bool(
    os.getenv("PYROVELOCITY_UPLOAD_RESULTS", "False")
)


if PYROVELOCITY_TESTING_FLAG:
    NUMBER_POSTERIOR_SAMPLES: int = 4
    MAX_EPOCHS: int = 300
    SUBSET_OBS: bool = True
    SUBSET_VARS: bool = True
else:
    NUMBER_POSTERIOR_SAMPLES: int = 30
    MAX_EPOCHS: int = 3000
    SUBSET_OBS: bool = False
    SUBSET_VARS: bool = False


@dataclass
class ResourcesJSON(DataClassJSONMixin):
    cpu: str
    mem: str
    gpu: str
    ephemeral_storage: str


testing_training_resource_requests = ResourcesJSON(
    cpu="4",
    mem="8Gi",
    gpu="1",
    ephemeral_storage="32Gi",
)

testing_training_resource_limits = ResourcesJSON(
    cpu="4",
    mem="8Gi",
    gpu="1",
    ephemeral_storage="32Gi",
)


default_training_resource_requests = ResourcesJSON(
    cpu="8",
    mem="30Gi",
    gpu="1",
    ephemeral_storage="50Gi",
)

default_training_resource_limits = ResourcesJSON(
    cpu="16",
    mem="60Gi",
    gpu="1",
    ephemeral_storage="200Gi",
)

large_training_resource_requests = ResourcesJSON(
    cpu="16",
    mem="120Gi",
    gpu="1",
    ephemeral_storage="50Gi",
)

large_training_resource_limits = ResourcesJSON(
    cpu="32",
    mem="200Gi",
    gpu="1",
    ephemeral_storage="200Gi",
)

default_resource_requests = ResourcesJSON(
    cpu="8",
    mem="30Gi",
    gpu="0",
    ephemeral_storage="50Gi",
)

default_resource_limits = ResourcesJSON(
    cpu="16",
    mem="60Gi",
    gpu="0",
    ephemeral_storage="200Gi",
)

medium_resource_requests = ResourcesJSON(
    cpu="16",
    mem="60Gi",
    gpu="0",
    ephemeral_storage="50Gi",
)

medium_resource_limits = ResourcesJSON(
    cpu="32",
    mem="120Gi",
    gpu="0",
    ephemeral_storage="200Gi",
)

large_resource_requests = ResourcesJSON(
    cpu="16",
    mem="120Gi",
    gpu="0",
    ephemeral_storage="50Gi",
)

large_resource_limits = ResourcesJSON(
    cpu="32",
    mem="200Gi",
    gpu="0",
    ephemeral_storage="200Gi",
)


@dataclass
class PostprocessConfiguration(DataClassJSONMixin):
    number_posterior_samples: int = field(
        default_factory=lambda: NUMBER_POSTERIOR_SAMPLES
    )


@dataclass
class WorkflowConfiguration(DataClassJSONMixin):
    download_dataset: DownloadDatasetInterface
    preprocess_data: PreprocessDataInterface
    training_configuration_1: PyroVelocityTrainInterface
    training_configuration_2: PyroVelocityTrainInterface
    postprocess_configuration: PostprocessConfiguration
    training_resources_requests: ResourcesJSON
    training_resources_limits: ResourcesJSON
    postprocessing_resources_requests: ResourcesJSON
    postprocessing_resources_limits: ResourcesJSON
    summarizing_resources_requests: ResourcesJSON
    summarizing_resources_limits: ResourcesJSON
    upload_results: bool = PYROVELOCITY_UPLOAD_RESULTS


@dataclass
class TrainingOutputs(DataClassJSONMixin):
    data_model: str
    data_model_path: str
    trained_data_path: FlyteFile
    model_path: FlyteDirectory
    posterior_samples_path: FlyteFile
    metrics_path: FlyteFile
    run_info_path: FlyteFile
    loss_plot_path: FlyteFile


@dataclass
class PostprocessOutputs(DataClassJSONMixin):
    pyrovelocity_data: FlyteFile
    postprocessed_data: FlyteFile


@dataclass
class SummarizeOutputs(DataClassJSONMixin):
    data_model_reports: FlyteDirectory
    dataframe: FlyteFile


simulated_dataset_args = DownloadDatasetInterface(
    data_set_name="simulated",
    source="simulate",
    n_obs=3000,
    n_vars=2000,
)
simulated_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    adata=f"{simulated_dataset_args.data_external_path}/{simulated_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="leiden",
)
simulated_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{simulated_preprocess_data_args.data_processed_path}/{simulated_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
simulated_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{simulated_preprocess_data_args.data_processed_path}/{simulated_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
simulated_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
simulated_configuration = WorkflowConfiguration(
    download_dataset=simulated_dataset_args,
    preprocess_data=simulated_preprocess_data_args,
    training_configuration_1=simulated_train_model1_args,
    training_configuration_2=simulated_train_model2_args,
    postprocess_configuration=simulated_postprocess_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=default_resource_requests,
    postprocessing_resources_limits=default_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)

pancreas_dataset_args = DownloadDatasetInterface(
    data_set_name="pancreas",
)
pancreas_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pancreas_dataset_args.data_set_name}",
    adata=f"{pancreas_dataset_args.data_external_path}/{pancreas_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    process_cytotrace=True,
)
pancreas_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{pancreas_preprocess_data_args.data_processed_path}/{pancreas_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pancreas_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
pancreas_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{pancreas_preprocess_data_args.data_processed_path}/{pancreas_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pancreas_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
pancreas_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
pancreas_configuration = WorkflowConfiguration(
    download_dataset=pancreas_dataset_args,
    preprocess_data=pancreas_preprocess_data_args,
    training_configuration_1=pancreas_train_model1_args,
    training_configuration_2=pancreas_train_model2_args,
    postprocess_configuration=pancreas_postprocess_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=medium_resource_requests,
    postprocessing_resources_limits=medium_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)

pbmc68k_dataset_args = DownloadDatasetInterface(
    data_set_name="pbmc68k",
)
pbmc68k_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pbmc68k_dataset_args.data_set_name}",
    adata=f"{pbmc68k_dataset_args.data_external_path}/{pbmc68k_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    default_velocity_mode="stochastic",
    cell_state="celltype",
    vector_field_basis="tsne",
)
pbmc68k_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{pbmc68k_preprocess_data_args.data_processed_path}/{pbmc68k_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pbmc68k_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
pbmc68k_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{pbmc68k_preprocess_data_args.data_processed_path}/{pbmc68k_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pbmc68k_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
pbmc68k_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
pbmc68k_configuration = WorkflowConfiguration(
    download_dataset=pbmc68k_dataset_args,
    preprocess_data=pbmc68k_preprocess_data_args,
    training_configuration_1=pbmc68k_train_model1_args,
    training_configuration_2=pbmc68k_train_model2_args,
    postprocess_configuration=pbmc68k_postprocess_configuration,
    training_resources_requests=large_training_resource_requests,
    training_resources_limits=large_training_resource_limits,
    postprocessing_resources_requests=large_resource_requests,
    postprocessing_resources_limits=large_resource_limits,
    summarizing_resources_requests=large_resource_requests,
    summarizing_resources_limits=large_resource_limits,
)


pons_dataset_args = DownloadDatasetInterface(
    data_set_name="pons",
)
pons_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pons_dataset_args.data_set_name}",
    adata=f"{pons_dataset_args.data_external_path}/{pons_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="celltype",
)
pons_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{pons_preprocess_data_args.data_processed_path}/{pons_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pons_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
pons_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{pons_preprocess_data_args.data_processed_path}/{pons_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pons_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
pons_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
pons_configuration = WorkflowConfiguration(
    download_dataset=pons_dataset_args,
    preprocess_data=pons_preprocess_data_args,
    training_configuration_1=pons_train_model1_args,
    training_configuration_2=pons_train_model2_args,
    postprocess_configuration=pons_postprocess_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=medium_resource_requests,
    postprocessing_resources_limits=medium_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)


larry_dataset_args = DownloadDatasetInterface(
    data_set_name="larry",
)
larry_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{larry_dataset_args.data_set_name}",
    adata=f"{larry_dataset_args.data_external_path}/{larry_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="state_info",
    vector_field_basis="emb",
)
larry_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{larry_preprocess_data_args.data_processed_path}/{larry_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    # svi_train=True,
    batch_size=4000,
    offset=False,
    max_epochs=MAX_EPOCHS,
)
larry_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{larry_preprocess_data_args.data_processed_path}/{larry_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_dataset_args.data_set_name}",
    model_identifier="model2",
    # svi_train=True,
    batch_size=4000,
    max_epochs=MAX_EPOCHS,
)
larry_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
larry_configuration = WorkflowConfiguration(
    download_dataset=larry_dataset_args,
    preprocess_data=larry_preprocess_data_args,
    training_configuration_1=larry_train_model1_args,
    training_configuration_2=larry_train_model2_args,
    postprocess_configuration=larry_postprocess_configuration,
    training_resources_requests=large_training_resource_requests,
    training_resources_limits=large_training_resource_limits,
    postprocessing_resources_requests=large_resource_requests,
    postprocessing_resources_limits=large_resource_limits,
    summarizing_resources_requests=large_resource_requests,
    summarizing_resources_limits=large_resource_limits,
)


if __name__ == "__main__":
    pass
