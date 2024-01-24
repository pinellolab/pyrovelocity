from dataclasses import asdict, make_dataclass
from datetime import timedelta
from typing import Any, Dict, Tuple, Type

from flytekit import Resources, task, workflow
from flytekit.extras.accelerators import T4
from flytekit.types.file import FlyteFile
from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.data import download_dataset
from pyrovelocity.logging import configure_logging
from pyrovelocity.workflows.configuration import create_dataclass_from_callable

logger = configure_logging(__name__)

custom_types_defaults: Dict[str, Tuple[Type, Any]] = {
    "data_set_name": (str, "pancreas"),
    "download_file_name": (str, "pancreas"),
    "source": (str, "scvelo"),
}

download_dataset_fields = create_dataclass_from_callable(
    download_dataset, custom_types_defaults
)

DownloadDatasetInterface = make_dataclass(
    "DownloadDatasetInterface",
    download_dataset_fields,
    bases=(DataClassJSONMixin,),
)
DownloadDatasetInterface.__module__ = __name__


@task(
    cache=True,
    cache_version="0.2.0b7",
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=20),
    requests=Resources(cpu="2", mem="4Gi", ephemeral_storage="16Gi"),
)
def download_data(download_dataset_args: DownloadDatasetInterface) -> FlyteFile:
    """
    Download external data.
    """
    # import time

    # time.sleep(7200)
    dataset_path = download_dataset(**asdict(download_dataset_args))
    return FlyteFile(path=dataset_path)


@workflow
def module_workflow(
    download_dataset_args: DownloadDatasetInterface = DownloadDatasetInterface(),
) -> FlyteFile:
    """
    Put all of the steps together into a single workflow.
    """
    data = download_data(download_dataset_args=download_dataset_args)
    return data


@workflow
def training_workflow() -> Tuple[FlyteFile, FlyteFile]:
    """
    Apply the module_workflow to all datasets.
    """
    simulated_data = module_workflow(
        download_dataset_args=DownloadDatasetInterface(
            data_set_name="simulated",
            download_file_name="simulated",
            source="simulate",
        )
    )

    pancreas_data = module_workflow(
        download_dataset_args=DownloadDatasetInterface(
            data_set_name="pancreas",
            download_file_name="pancreas",
            source="scvelo",
        )
    )
    return (simulated_data, pancreas_data)


if __name__ == "__main__":
    print(f"Running module_workflow() { module_workflow() }")
    print(f"Running training_workflow() { training_workflow() }")
