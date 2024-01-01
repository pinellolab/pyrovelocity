import os

from components.environment_log_component import (
    create_environment_log_component,
)
from construct_pipeline import create_complete_pipeline
from dotenv import load_dotenv
from google.cloud import aiplatform
from hydra_zen import (
    make_custom_builds_fn,
    store,
    zen,
)

load_dotenv(".envrc")


base_image = os.getenv("ARGO_PIPELINE_BASE_IMAGE")
pipeline_root = os.getenv("ARGO_PIPELINE_ROOT")

builds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

EnvironmentLogComponentConf = builds(create_environment_log_component)
base_envlog = EnvironmentLogComponentConf(
    base_image=base_image,
    display_name="environment_log_component",
    machine_type=os.environ["ARGO_ENVIRONMENT_LOG_MACHINE_TYPE"],
    accelerator_type=os.environ["ARGO_ACCELERATOR_TYPE"],
    accelerator_count=int(os.environ["ARGO_ACCELERATOR_COUNT"]),
)
envlog_store = store(group="job/environment_log_component")
envlog_store(base_envlog, name="base_for_envlog_component")


PipelineConf = builds(
    create_complete_pipeline,
)
base_pipeline = PipelineConf(
    pipeline_root=pipeline_root,
    environment_log_component=base_envlog,
)

pipeline_store = store(group="job/pipeline")
pipeline_store(base_pipeline, name="base_pipeline")

aiplatform.pipeline_jobs = aiplatform.PipelineJob
JobConf = builds(
    aiplatform.PipelineJob.from_pipeline_func,
)
base_job = JobConf(
    pipeline_func=base_pipeline,
    parameter_values={
        "project": os.environ["ARGO_GCP_PROJECT_ID"],
        "location": os.environ["ARGO_GCP_REGION"],
        "message": "message text",
    },
)

job_store = store(group="job")
job_store(base_job, name="base_job")


@store(
    name="distributed_pipeline", hydra_defaults=["_self_", {"job": "base_job"}]
)
def task_function(job):
    print("submitting pipeline")
    print(job)
    job.submit()


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(task_function).hydra_main(
        config_name="distributed_pipeline",
        version_base="1.1",
        config_path=".",
    )
