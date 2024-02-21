from dataclasses import make_dataclass
from typing import Any, Dict, Tuple, Type

from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.data import download_dataset
from pyrovelocity.preprocess import preprocess_dataset
from pyrovelocity.train import train_dataset
from pyrovelocity.workflows.configuration import create_dataclass_from_callable

__all__ = [
    "DownloadDatasetInterface",
    "PreprocessDataInterface",
    "PyroVelocityTrainInterface",
]

# These can be used to override the default values of the dataclass
# but are generally only required if the callable interface lacks
# complete and dataclass-compatible type annotations.
custom_types_defaults: Dict[str, Tuple[Type, Any]] = {
    # "data_set_name": (str, "pancreas"),
    # "source": (str, "pyrovelocity"),
    "data_external_path": (str, "data/external"),
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


preprocess_data_types_defaults: Dict[str, Tuple[Type, Any]] = {
    "adata": (str, "data/external/simulated.h5ad"),
    "data_processed_path": (str, "data/processed"),
}

preprocess_data_fields = create_dataclass_from_callable(
    preprocess_dataset,
    preprocess_data_types_defaults,
)

PreprocessDataInterface = make_dataclass(
    "PreprocessDataInterface",
    preprocess_data_fields,
    bases=(DataClassJSONMixin,),
)
PreprocessDataInterface.__module__ = __name__

pyrovelocity_train_types_defaults: Dict[str, Tuple[Type, Any]] = {
    "adata": (str, "data/processed/simulated_processed.h5ad"),
    "use_gpu": (bool, False),
}

pyrovelocity_train_fields = create_dataclass_from_callable(
    train_dataset,
    pyrovelocity_train_types_defaults,
)

PyroVelocityTrainInterface = make_dataclass(
    "PyroVelocityTrainInterface",
    pyrovelocity_train_fields,
    bases=(DataClassJSONMixin,),
)
PyroVelocityTrainInterface.__module__ = __name__
