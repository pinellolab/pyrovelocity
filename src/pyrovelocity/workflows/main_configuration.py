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
from pyrovelocity.workflows.constants import (
    PYROVELOCITY_TESTING_FLAG,
    PYROVELOCITY_UPLOAD_RESULTS,
)

__all__ = [
    "ResourcesJSON",
    "PostprocessConfiguration",
    "PreprocessOutputs",
    "TrainingOutputs",
    "PostprocessOutputs",
    "SummarizeOutputs",
    "WorkflowConfiguration",
]


logger = configure_logging(__name__)


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

extra_large_training_resource_requests = ResourcesJSON(
    cpu="20",
    mem="160Gi",
    gpu="1",
    ephemeral_storage="100Gi",
)

extra_large_training_resource_limits = ResourcesJSON(
    cpu="46",
    mem="250Gi",
    gpu="1",
    ephemeral_storage="400Gi",
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
class SummarizeConfiguration(DataClassJSONMixin):
    selected_genes: list[str] = field(default_factory=lambda: [""])


@dataclass
class WorkflowConfiguration(DataClassJSONMixin):
    download_dataset: DownloadDatasetInterface
    preprocess_data: PreprocessDataInterface
    training_configuration_1: PyroVelocityTrainInterface
    training_configuration_2: PyroVelocityTrainInterface
    postprocess_configuration: PostprocessConfiguration
    summarize_configuration: SummarizeConfiguration
    training_resources_requests: ResourcesJSON
    training_resources_limits: ResourcesJSON
    postprocessing_resources_requests: ResourcesJSON
    postprocessing_resources_limits: ResourcesJSON
    summarizing_resources_requests: ResourcesJSON
    summarizing_resources_limits: ResourcesJSON
    accelerator_type: str = "nvidia-tesla-t4"
    upload_results: bool = PYROVELOCITY_UPLOAD_RESULTS


@dataclass
class PreprocessOutputs(DataClassJSONMixin):
    processed_data: FlyteFile
    processed_reports: FlyteDirectory


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
    loss_csv_path: FlyteFile


@dataclass
class PostprocessOutputs(DataClassJSONMixin):
    pyrovelocity_data: FlyteFile
    postprocessed_data: FlyteFile
    metrics_path: FlyteFile


@dataclass
class SummarizeOutputs(DataClassJSONMixin):
    data_model: str
    data_model_reports: FlyteDirectory
    dataframe: FlyteFile
    run_metrics_path: FlyteFile
    run_info_path: FlyteFile
    loss_plot_path: FlyteFile
    loss_csv_path: FlyteFile
    combined_metrics_path: FlyteFile
    pyrovelocity_data: FlyteFile
    postprocessed_data: FlyteFile


@dataclass
class CombinedMetricsOutputs(DataClassJSONMixin):
    metrics_json: FlyteFile
    metrics_latex: FlyteFile
    metrics_html: FlyteFile
    metrics_md: FlyteFile
    elbo_latex: FlyteFile
    elbo_html: FlyteFile
    elbo_md: FlyteFile
    mae_latex: FlyteFile
    mae_html: FlyteFile
    mae_md: FlyteFile


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
simulated_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "150",
        "366",
        "1853",
        "1157",
        "804",
        "360",
    ]
)
simulated_configuration = WorkflowConfiguration(
    download_dataset=simulated_dataset_args,
    preprocess_data=simulated_preprocess_data_args,
    training_configuration_1=simulated_train_model1_args,
    training_configuration_2=simulated_train_model2_args,
    postprocess_configuration=simulated_postprocess_configuration,
    summarize_configuration=simulated_summary_configuration,
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
pancreas_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "Cpe",
        "Ins2",
        "Cck",
        "Ttr",
        "Krt7",
        "Spp1",
    ]
)
pancreas_configuration = WorkflowConfiguration(
    download_dataset=pancreas_dataset_args,
    preprocess_data=pancreas_preprocess_data_args,
    training_configuration_1=pancreas_train_model1_args,
    training_configuration_2=pancreas_train_model2_args,
    postprocess_configuration=pancreas_postprocess_configuration,
    summarize_configuration=pancreas_summary_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=medium_resource_requests,
    postprocessing_resources_limits=medium_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)


bonemarrow_dataset_args = DownloadDatasetInterface(
    data_set_name="bonemarrow",
)
bonemarrow_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{bonemarrow_dataset_args.data_set_name}",
    adata=f"{bonemarrow_dataset_args.data_external_path}/{bonemarrow_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
)
bonemarrow_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{bonemarrow_preprocess_data_args.data_processed_path}/{bonemarrow_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{bonemarrow_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
bonemarrow_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{bonemarrow_preprocess_data_args.data_processed_path}/{bonemarrow_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{bonemarrow_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
bonemarrow_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
bonemarrow_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "ELANE",
        "HDAC7",
        "CITED2",
        "SLC40A1",
        "VPREB1",
        "MYB",
    ]
)
bonemarrow_configuration = WorkflowConfiguration(
    download_dataset=bonemarrow_dataset_args,
    preprocess_data=bonemarrow_preprocess_data_args,
    training_configuration_1=bonemarrow_train_model1_args,
    training_configuration_2=bonemarrow_train_model2_args,
    postprocess_configuration=bonemarrow_postprocess_configuration,
    summarize_configuration=bonemarrow_summary_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=medium_resource_requests,
    postprocessing_resources_limits=medium_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)


pbmc5k_dataset_args = DownloadDatasetInterface(
    data_set_name="pbmc5k",
)
pbmc5k_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pbmc5k_dataset_args.data_set_name}",
    adata=f"{pbmc5k_dataset_args.data_external_path}/{pbmc5k_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="celltype",
)
pbmc5k_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{pbmc5k_preprocess_data_args.data_processed_path}/{pbmc5k_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pbmc5k_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
pbmc5k_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{pbmc5k_preprocess_data_args.data_processed_path}/{pbmc5k_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pbmc5k_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
pbmc5k_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
pbmc5k_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "LYZ",
        "S100A9",
        "IGHM",
        "HLA-DQA1",
        "MS4A1",
        "IL32",
    ]
)
pbmc5k_configuration = WorkflowConfiguration(
    download_dataset=pbmc5k_dataset_args,
    preprocess_data=pbmc5k_preprocess_data_args,
    training_configuration_1=pbmc5k_train_model1_args,
    training_configuration_2=pbmc5k_train_model2_args,
    postprocess_configuration=pbmc5k_postprocess_configuration,
    summarize_configuration=pbmc5k_summary_configuration,
    training_resources_requests=large_training_resource_requests,
    training_resources_limits=large_training_resource_limits,
    postprocessing_resources_requests=large_resource_requests,
    postprocessing_resources_limits=large_resource_limits,
    summarizing_resources_requests=large_resource_requests,
    summarizing_resources_limits=large_resource_limits,
)


pbmc10k_dataset_args = DownloadDatasetInterface(
    data_set_name="pbmc10k",
)
pbmc10k_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{pbmc10k_dataset_args.data_set_name}",
    adata=f"{pbmc10k_dataset_args.data_external_path}/{pbmc10k_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="celltype",
)
pbmc10k_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{pbmc10k_preprocess_data_args.data_processed_path}/{pbmc10k_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pbmc10k_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
pbmc10k_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{pbmc10k_preprocess_data_args.data_processed_path}/{pbmc10k_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{pbmc10k_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
pbmc10k_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
pbmc10k_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "LYZ",
        "S100A9",
        "IGHM",
        "HLA-DQA1",
        "MS4A1",
        "IL32",
    ]
)
pbmc10k_configuration = WorkflowConfiguration(
    download_dataset=pbmc10k_dataset_args,
    preprocess_data=pbmc10k_preprocess_data_args,
    training_configuration_1=pbmc10k_train_model1_args,
    training_configuration_2=pbmc10k_train_model2_args,
    postprocess_configuration=pbmc10k_postprocess_configuration,
    summarize_configuration=pbmc10k_summary_configuration,
    training_resources_requests=large_training_resource_requests,
    training_resources_limits=large_training_resource_limits,
    postprocessing_resources_requests=large_resource_requests,
    postprocessing_resources_limits=large_resource_limits,
    summarizing_resources_requests=large_resource_requests,
    summarizing_resources_limits=large_resource_limits,
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
pbmc68k_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "TPT1",
        "EEF1A1",
        "LTB",
        "GNLY",
        "TMSB4X",
        "FTL",
    ]
)
pbmc68k_configuration = WorkflowConfiguration(
    download_dataset=pbmc68k_dataset_args,
    preprocess_data=pbmc68k_preprocess_data_args,
    training_configuration_1=pbmc68k_train_model1_args,
    training_configuration_2=pbmc68k_train_model2_args,
    postprocess_configuration=pbmc68k_postprocess_configuration,
    summarize_configuration=pbmc68k_summary_configuration,
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
pons_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "Mag",
        "Cldn11",
        "Cntn1",
        "Marcks",
        "Tubb4a",
        "Mbp",
    ]
)
pons_configuration = WorkflowConfiguration(
    download_dataset=pons_dataset_args,
    preprocess_data=pons_preprocess_data_args,
    training_configuration_1=pons_train_model1_args,
    training_configuration_2=pons_train_model2_args,
    postprocess_configuration=pons_postprocess_configuration,
    summarize_configuration=pons_summary_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=medium_resource_requests,
    postprocessing_resources_limits=medium_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)

larry_neu_dataset_args = DownloadDatasetInterface(
    data_set_name="larry_neu",
)
larry_neu_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{larry_neu_dataset_args.data_set_name}",
    adata=f"{larry_neu_dataset_args.data_external_path}/{larry_neu_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="state_info",
    vector_field_basis="emb",
)
larry_neu_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{larry_neu_preprocess_data_args.data_processed_path}/{larry_neu_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_neu_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
larry_neu_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{larry_neu_preprocess_data_args.data_processed_path}/{larry_neu_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_neu_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
larry_neu_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
larry_neu_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "S100a8",
        "S100a9",
        "Ngp",
        "Lilrb4",
        "Mpp1",
        "Srgn",
    ]
)
larry_neu_configuration = WorkflowConfiguration(
    download_dataset=larry_neu_dataset_args,
    preprocess_data=larry_neu_preprocess_data_args,
    training_configuration_1=larry_neu_train_model1_args,
    training_configuration_2=larry_neu_train_model2_args,
    postprocess_configuration=larry_neu_postprocess_configuration,
    summarize_configuration=larry_neu_summary_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=medium_resource_requests,
    postprocessing_resources_limits=medium_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)

larry_mono_dataset_args = DownloadDatasetInterface(
    data_set_name="larry_mono",
)
larry_mono_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{larry_mono_dataset_args.data_set_name}",
    adata=f"{larry_mono_dataset_args.data_external_path}/{larry_mono_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="state_info",
    vector_field_basis="emb",
)
larry_mono_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{larry_mono_preprocess_data_args.data_processed_path}/{larry_mono_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_mono_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
larry_mono_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{larry_mono_preprocess_data_args.data_processed_path}/{larry_mono_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_mono_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
larry_mono_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
larry_mono_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "Vim",
        "Fcer1g",
        "Fth1",
        "Ctsc",
        "Itm2b",
        "Sell",
    ]
)
larry_mono_configuration = WorkflowConfiguration(
    download_dataset=larry_mono_dataset_args,
    preprocess_data=larry_mono_preprocess_data_args,
    training_configuration_1=larry_mono_train_model1_args,
    training_configuration_2=larry_mono_train_model2_args,
    postprocess_configuration=larry_mono_postprocess_configuration,
    summarize_configuration=larry_mono_summary_configuration,
    training_resources_requests=default_training_resource_requests,
    training_resources_limits=default_training_resource_limits,
    postprocessing_resources_requests=medium_resource_requests,
    postprocessing_resources_limits=medium_resource_limits,
    summarizing_resources_requests=default_resource_requests,
    summarizing_resources_limits=default_resource_limits,
)

larry_multilineage_dataset_args = DownloadDatasetInterface(
    data_set_name="larry_multilineage",
)
larry_multilineage_preprocess_data_args = PreprocessDataInterface(
    data_set_name=f"{larry_multilineage_dataset_args.data_set_name}",
    adata=f"{larry_multilineage_dataset_args.data_external_path}/{larry_multilineage_dataset_args.data_set_name}.h5ad",
    use_obs_subset=SUBSET_OBS,
    use_vars_subset=SUBSET_VARS,
    cell_state="state_info",
    vector_field_basis="emb",
)
larry_multilineage_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{larry_multilineage_preprocess_data_args.data_processed_path}/{larry_multilineage_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_multilineage_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
larry_multilineage_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{larry_multilineage_preprocess_data_args.data_processed_path}/{larry_multilineage_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_multilineage_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
larry_multilineage_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
larry_multilineage_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "Itgb2",
        "S100a9",
        "Fcer1g",
        "Lilrb4",
        "Vim",
        "Serbp1",
    ]
)
larry_multilineage_configuration = WorkflowConfiguration(
    download_dataset=larry_multilineage_dataset_args,
    preprocess_data=larry_multilineage_preprocess_data_args,
    training_configuration_1=larry_multilineage_train_model1_args,
    training_configuration_2=larry_multilineage_train_model2_args,
    postprocess_configuration=larry_multilineage_postprocess_configuration,
    summarize_configuration=larry_multilineage_summary_configuration,
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
# To train the model with batching, set the batch_size argument
# e.g., batch_size=4000.
larry_train_model1_args = PyroVelocityTrainInterface(
    adata=f"{larry_preprocess_data_args.data_processed_path}/{larry_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_dataset_args.data_set_name}",
    model_identifier="model1",
    guide_type="auto_t0_constraint",
    offset=False,
    max_epochs=MAX_EPOCHS,
)
larry_train_model2_args = PyroVelocityTrainInterface(
    adata=f"{larry_preprocess_data_args.data_processed_path}/{larry_dataset_args.data_set_name}_processed.h5ad",
    data_set_name=f"{larry_dataset_args.data_set_name}",
    model_identifier="model2",
    max_epochs=MAX_EPOCHS,
)
larry_postprocess_configuration = PostprocessConfiguration(
    number_posterior_samples=NUMBER_POSTERIOR_SAMPLES,
)
larry_summary_configuration = SummarizeConfiguration(
    selected_genes=[
        "Itgb2",
        "S100a8",
        "Lyz2",
        "Fcer1g",
        "Csf2rb",
        "Ms4a3",
    ]
)
larry_accelerator_type = "nvidia-tesla-a100"
larry_configuration = WorkflowConfiguration(
    download_dataset=larry_dataset_args,
    preprocess_data=larry_preprocess_data_args,
    training_configuration_1=larry_train_model1_args,
    training_configuration_2=larry_train_model2_args,
    postprocess_configuration=larry_postprocess_configuration,
    summarize_configuration=larry_summary_configuration,
    training_resources_requests=extra_large_training_resource_requests,
    training_resources_limits=extra_large_training_resource_limits,
    postprocessing_resources_requests=large_resource_requests,
    postprocessing_resources_limits=large_resource_limits,
    summarizing_resources_requests=large_resource_requests,
    summarizing_resources_limits=large_resource_limits,
    accelerator_type=larry_accelerator_type,
)


if __name__ == "__main__":
    pass
