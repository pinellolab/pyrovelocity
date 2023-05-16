import os
from typing import Callable

from dotenv import load_dotenv
from hydra_zen import make_custom_builds_fn
from kfp.v2 import dsl
from kfp.v2.dsl import Dataset  # Output, Input, Model, Artifact


load_dotenv(".envrc")


base_image = os.getenv("ARGO_PIPELINE_BASE_IMAGE")
pipeline_root = os.getenv("ARGO_PIPELINE_ROOT")

builds = make_custom_builds_fn(populate_full_signature=True)


def create_complete_pipeline(
    pipeline_root: str,
    environment_log_component: Callable,
):
    @dsl.pipeline(
        name="complete pipeline",
        description="complete run of pipeline",
        pipeline_root=pipeline_root,
    )
    def complete_pipeline(
        project: str,
        location: str,
        message: str,
    ) -> Dataset:
        return environment_log_component(
            project=project,
            location=location,
            message=message,
        ).outputs["environment_info"]

    return complete_pipeline
