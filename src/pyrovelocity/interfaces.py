from dataclasses import make_dataclass
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

from mashumaro.mixins.json import DataClassJSONMixin

from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.workflows.configuration import create_dataclass_from_callable


__all__ = [
    "DownloadDatasetInterface",
    "PreprocessDataInterface",
    "PyroVelocityTrainInterface",
]

if TYPE_CHECKING:
    # The following is not executed at runtime and is only intended to prevent
    # interpretation of dataclasses as variables
    #
    #   (variable) DownloadDatasetInterface: type
    #
    # in favor of interpreting dataclasses as
    #
    #   (class) DownloadDatasetInterface
    #
    # in IDEs that support type hinting/checking.
    class DownloadDatasetInterface(DataClassJSONMixin):
        pass

    class PreprocessDataInterface(DataClassJSONMixin):
        pass

    class PyroVelocityTrainInterface(DataClassJSONMixin):
        pass


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
    # "use_gpu": (str, "auto"),
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


class AcceleratorType(Enum):
    """
    Enumeration of accelerator types denoting a subset of the accelerators recognized
    by PyTorch Lightning's AcceleratorRegistry.

    See also:
    - https://lightning.ai/docs/pytorch/2.1.4/extensions/accelerator.html
    - https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.4/src/lightning/pytorch/trainer/connectors/accelerator_connector.py#L209-L217

    This enumeration supports validation of the selection of accelerator types for
    computations, providing a straightforward way to specify the desired
    computation device. It accounts for a subset of the available accelerators
    as reported by `AcceleratorRegistry.available_accelerators()` from PyTorch
    Lightning, specifically targeting 'cpu' and 'cuda' accelerators, while also
    offering an 'auto' option for automatic selection.

    It would be preferable to restrict this to subclasses of
    `lightning.pytorch.accelerators.Accelerator`, but there is no such instance
    that supporting the "auto" option, which lightning only references via a
    string.
    https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.4/src/lightning/pytorch/accelerators/accelerator.py.

    Members:
    - AUTO: Represents an automatic choice of accelerator based on the system's
            availability and configuration. This option is intended to provide
            flexibility, allowing PyTorch Lightning to automatically select the
            most appropriate accelerator: usually CPU vs CUDA.
    - CPU: Specifies the use of the CPU for computations. This is a universal
           option, available on all systems.
           See https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.4/src/lightning/pytorch/accelerators/cpu.py.
    - CUDA: Specifies the use of NVIDIA CUDA-enabled GPUs for computations. This
            option should be selected when intending to leverage GPU acceleration,
            assuming CUDA-compatible hardware is available.
            See https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.4/src/lightning/pytorch/accelerators/cuda.py.

    This enumeration does not cover all the accelerators available in
    PyTorch Lightning, such as 'tpu' and 'mps'. It reflects options that are
    tested with this library.
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
