import copy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any
from typing import List
from typing import Type

from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.interfaces import DownloadDatasetInterface
from pyrovelocity.interfaces import PreprocessDataInterface
from pyrovelocity.interfaces import PyroVelocityTrainInterface
from pyrovelocity.logging import configure_logging


__all__ = ["ResourcesJSON", "TrainingOutputs", "WorkflowConfiguration"]


logger = configure_logging(__name__)

CONSTANT_NUMBER_POSTERIOR_SAMPLES: int = 3
UNIFORM_MAX_EPOCHS = 300


@dataclass
class ResourcesJSON(DataClassJSONMixin):
    cpu: str
    mem: str
    gpu: str
    ephemeral_storage: str


testing_training_resources = [
    ResourcesJSON(
        cpu="4",
        mem="8Gi",
        gpu="1",
        ephemeral_storage="32Gi",
    ),
    ResourcesJSON(
        cpu="4",
        mem="8Gi",
        gpu="1",
        ephemeral_storage="32Gi",
    ),
]

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

default_training_resources = [
    ResourcesJSON(
        cpu="8",
        mem="30Gi",
        gpu="1",
        ephemeral_storage="50Gi",
    ),
    ResourcesJSON(
        cpu="16",
        mem="60Gi",
        gpu="1",
        ephemeral_storage="200Gi",
    ),
]

large_training_resources = [
    ResourcesJSON(
        cpu="16",
        mem="120Gi",
        gpu="1",
        ephemeral_storage="50Gi",
    ),
    ResourcesJSON(
        cpu="32",
        mem="200Gi",
        gpu="1",
        ephemeral_storage="200Gi",
    ),
]

default_resources = [
    ResourcesJSON(
        cpu="8",
        mem="30Gi",
        gpu="0",
        ephemeral_storage="50Gi",
    ),
    ResourcesJSON(
        cpu="16",
        mem="60Gi",
        gpu="0",
        ephemeral_storage="200Gi",
    ),
]

large_resources = [
    ResourcesJSON(
        cpu="16",
        mem="120Gi",
        gpu="0",
        ephemeral_storage="50Gi",
    ),
    ResourcesJSON(
        cpu="32",
        mem="200Gi",
        gpu="0",
        ephemeral_storage="200Gi",
    ),
]


@dataclass
class PostprocessConfiguration(DataClassJSONMixin):
    number_posterior_samples: int = field(
        default_factory=lambda: CONSTANT_NUMBER_POSTERIOR_SAMPLES
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


simulated_dataset_args = DownloadDatasetInterface(
    data_set_name="simulated",
    source="simulate",
    n_obs=3000,
    n_vars=2000,
)
simulated_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    adata=f"{simulated_dataset_args.data_external_path}/{simulated_dataset_args.data_set_name}.h5ad",
    cell_state="leiden",
)
simulated_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{simulated_preprocess_data_args.data_processed_path}/{simulated_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=UNIFORM_MAX_EPOCHS,
)
simulated_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{simulated_preprocess_data_args.data_processed_path}/{simulated_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{simulated_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=UNIFORM_MAX_EPOCHS,
)
simulated_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=CONSTANT_NUMBER_POSTERIOR_SAMPLES,
)
simulated_configuration = WorkflowConfiguration(
    download_dataset=simulated_dataset_args,
    preprocess_data=simulated_preprocess_data_args,
    training_configuration_1=simulated_train_model1_args,
    training_configuration_2=simulated_train_model2_args,
    postprocess_configuration=simulated_postprocess_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=default_training_resource_requests,
    postprocessing_resources_limits=default_training_resource_limits,
)

# logger.debug(fields(simulated_configuration))

# test_simulated_configuration = copy.deepcopy(simulated_configuration)
# test_simulated_configuration.download_dataset.n_obs = 200
# test_simulated_configuration.download_dataset.n_vars = 100
# test_simulated_configuration.training_configurations[0].max_epochs = 300
# test_simulated_configuration.training_configurations[1].max_epochs = 300

# pancreas_dataset_args = DownloadDatasetInterface(
#     data_set_name="pancreas",
# )
# pancreas_preprocess_data_args = PreprocessDataInterface(
#     data_set_name=f"{pancreas_dataset_args.data_set_name}",
#     adata=f"{pancreas_dataset_args.data_external_path}/{pancreas_dataset_args.data_set_name}.h5ad",
#     process_cytotrace=True,
# )
# pancreas_train_model1_args = PyroVelocityTrainInterface(
#     adata=f"{pancreas_preprocess_data_args.data_processed_path}/{pancreas_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{pancreas_dataset_args.data_set_name}",
#     model_identifier="model1",
#     guide_type="auto_t0_constraint",
#     offset=False,
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )
# pancreas_train_model2_args = PyroVelocityTrainInterface(
#     adata=f"{pancreas_preprocess_data_args.data_processed_path}/{pancreas_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{pancreas_dataset_args.data_set_name}",
#     model_identifier="model2",
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )

# pancreas_configuration = WorkflowConfiguration(
#     download_dataset=pancreas_dataset_args,
#     preprocess_data=pancreas_preprocess_data_args,
#     training_configurations=[
#         pancreas_train_model1_args,
#         pancreas_train_model2_args,
#     ],
#     # training_resources=testing_training_resources,
#     training_resources=default_training_resources,
#     postprocessing_resources=default_resources,
#     number_posterior_samples=CONSTANT_NUMBER_POSTERIOR_SAMPLES,
# )

# # test_pancreas_configuration = copy.deepcopy(pancreas_configuration)
# # test_pancreas_configuration.download_dataset.n_obs = 200
# # test_pancreas_configuration.download_dataset.n_vars = 500
# # test_pancreas_configuration.training_configurations[0].max_epochs = 300
# # test_pancreas_configuration.training_configurations[1].max_epochs = 300

# pbmc68k_dataset_args = DownloadDatasetInterface(
#     data_set_name="pbmc68k",
#     # n_obs=500,
# )
# pbmc68k_preprocess_data_args = PreprocessDataInterface(
#     data_set_name=f"{pbmc68k_dataset_args.data_set_name}",
#     adata=f"{pbmc68k_dataset_args.data_external_path}/{pbmc68k_dataset_args.data_set_name}.h5ad",
#     default_velocity_mode="stochastic",
#     cell_state="celltype",
#     vector_field_basis="tsne",
# )
# pbmc68k_train_model1_args = PyroVelocityTrainInterface(
#     adata=f"{pbmc68k_preprocess_data_args.data_processed_path}/{pbmc68k_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{pbmc68k_dataset_args.data_set_name}",
#     model_identifier="model1",
#     guide_type="auto_t0_constraint",
#     offset=False,
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )
# pbmc68k_train_model2_args = PyroVelocityTrainInterface(
#     adata=f"{pbmc68k_preprocess_data_args.data_processed_path}/{pbmc68k_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{pbmc68k_dataset_args.data_set_name}",
#     model_identifier="model2",
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )

# pbmc68k_configuration = WorkflowConfiguration(
#     download_dataset=pbmc68k_dataset_args,
#     preprocess_data=pbmc68k_preprocess_data_args,
#     training_configurations=[
#         pbmc68k_train_model1_args,
#         pbmc68k_train_model2_args,
#     ],
#     # training_resources=testing_training_resources,
#     training_resources=default_training_resources,
#     postprocessing_resources=large_resources,
#     number_posterior_samples=CONSTANT_NUMBER_POSTERIOR_SAMPLES,
# )

# pons_dataset_args = DownloadDatasetInterface(
#     data_set_name="pons",
#     # n_obs=200,
#     # n_vars=500,
# )
# pons_preprocess_data_args = PreprocessDataInterface(
#     data_set_name=f"{pons_dataset_args.data_set_name}",
#     adata=f"{pons_dataset_args.data_external_path}/{pons_dataset_args.data_set_name}.h5ad",
#     cell_state="celltype",
# )
# pons_train_model1_args = PyroVelocityTrainInterface(
#     adata=f"{pons_preprocess_data_args.data_processed_path}/{pons_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{pons_dataset_args.data_set_name}",
#     model_identifier="model1",
#     guide_type="auto_t0_constraint",
#     offset=False,
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )
# pons_train_model2_args = PyroVelocityTrainInterface(
#     adata=f"{pons_preprocess_data_args.data_processed_path}/{pons_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{pons_dataset_args.data_set_name}",
#     model_identifier="model2",
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )

# pons_configuration = WorkflowConfiguration(
#     download_dataset=pons_dataset_args,
#     preprocess_data=pons_preprocess_data_args,
#     training_configurations=[
#         pons_train_model1_args,
#         pons_train_model2_args,
#     ],
#     # training_resources=testing_training_resources,
#     training_resources=default_training_resources,
#     postprocessing_resources=default_resources,
#     number_posterior_samples=CONSTANT_NUMBER_POSTERIOR_SAMPLES,
# )


# larry_dataset_args = DownloadDatasetInterface(
#     data_set_name="larry",
#     # n_obs=500,
#     # n_vars=2000,
# )
# larry_preprocess_data_args = PreprocessDataInterface(
#     data_set_name=f"{larry_dataset_args.data_set_name}",
#     adata=f"{larry_dataset_args.data_external_path}/{larry_dataset_args.data_set_name}.h5ad",
#     cell_state="state_info",
#     vector_field_basis="emb",
# )
# larry_train_model1_args = PyroVelocityTrainInterface(
#     adata=f"{larry_preprocess_data_args.data_processed_path}/{larry_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{larry_dataset_args.data_set_name}",
#     model_identifier="model1",
#     guide_type="auto_t0_constraint",
#     # svi_train=True,
#     batch_size=4000,
#     offset=False,
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )
# larry_train_model2_args = PyroVelocityTrainInterface(
#     adata=f"{larry_preprocess_data_args.data_processed_path}/{larry_dataset_args.data_set_name}_processed.h5ad",
#     data_set_name=f"{larry_dataset_args.data_set_name}",
#     model_identifier="model2",
#     # svi_train=True,
#     batch_size=4000,
#     max_epochs=UNIFORM_MAX_EPOCHS,
# )

# larry_configuration = WorkflowConfiguration(
#     download_dataset=larry_dataset_args,
#     preprocess_data=larry_preprocess_data_args,
#     training_configurations=[
#         larry_train_model1_args,
#         larry_train_model2_args,
#     ],
#     # training_resources=testing_training_resources,
#     training_resources=large_training_resources,
#     postprocessing_resources=large_resources,
#     number_posterior_samples=CONSTANT_NUMBER_POSTERIOR_SAMPLES,
# )


if __name__ == "__main__":
    pass
