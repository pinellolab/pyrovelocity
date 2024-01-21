from dataclasses import asdict, make_dataclass
from datetime import timedelta
from pprint import pformat
from typing import Any, Dict, Optional, Tuple, Type

import joblib
import pandas as pd
from flytekit import Resources, task, workflow
from flytekit.extras.accelerators import T4
from flytekit.types.file import JoblibSerializedFile
from mashumaro.mixins.json import DataClassJSONMixin
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression

from pyrovelocity.flytezen.configuration import create_dataclass_from_callable
from pyrovelocity.logging import configure_logging

logger = configure_logging("pyrovelocity.flytezen.workflows.main_workflow")

custom_types_defaults: Dict[str, Tuple[Type, Any]] = {
    "class_weight": (Optional[dict], None),
    "random_state": (Optional[int], None),
    "max_iter": (int, 2000),
    "n_jobs": (Optional[int], None),
    "l1_ratio": (Optional[float], None),
}

logistic_regression_fields = create_dataclass_from_callable(
    LogisticRegression, custom_types_defaults
)

LogisticRegressionInterface = make_dataclass(
    "LogisticRegressionInterface",
    logistic_regression_fields,
    bases=(DataClassJSONMixin,),
    # TODO: Python 3.12, https://github.com/python/cpython/pull/102104
    # module=__name__,
)
LogisticRegressionInterface.__module__ = __name__

sample_columns = [
    "alcohol",
    "target",
]

sample_data = [
    [13.0, 0],
    [14.0, 1],
    [12.5, 2],
]


@task(
    cache=True,
    cache_version="0.1.0",
    retries=3,
    interruptible=False,
    timeout=timedelta(minutes=20),
    container_image="{{.image.gpu.fqn}}:{{.image.gpu.version}}",
    requests=Resources(
        cpu="200m", mem="400Mi", ephemeral_storage="1Gi", gpu="1"
    ),
    accelerator=T4,
)
def get_data() -> pd.DataFrame:
    """
    Get the wine dataset.
    """
    # import time

    # time.sleep(7200)
    return load_wine(as_frame=True).frame


@task(
    cache=False,
    cache_version="0.1.0",
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=10),
    requests=Resources(cpu="200m", mem="400Mi", ephemeral_storage="1Gi"),
)
def process_data(
    data: pd.DataFrame = pd.DataFrame(data=sample_data, columns=sample_columns),
) -> pd.DataFrame:
    """
    Simplify the task from a 3-class to a binary classification problem.
    """
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


@task(
    cache=True,
    cache_version="0.1.0",
    retries=3,
    interruptible=True,
    timeout=timedelta(minutes=10),
    requests=Resources(cpu="200m", mem="400Mi", ephemeral_storage="1Gi"),
)
def train_model(
    data: pd.DataFrame = pd.DataFrame(data=sample_data, columns=sample_columns),
    logistic_regression: LogisticRegressionInterface = LogisticRegressionInterface(
        max_iter=1200
    ),
) -> JoblibSerializedFile:
    """
    Train a model on the wine dataset.
    """
    features = data.drop("target", axis="columns")
    target = data["target"]
    logger.info(f"{pformat(logistic_regression)}\n\n")
    model = LogisticRegression(**asdict(logistic_regression))
    model_path = "logistic_regression_model.joblib"
    joblib.dump(model, model_path)
    model_file = JoblibSerializedFile(model_path)
    return model_file


@workflow
def training_workflow(
    logistic_regression: LogisticRegressionInterface = LogisticRegressionInterface(
        max_iter=2000
    ),
) -> JoblibSerializedFile:
    """
    Put all of the steps together into a single workflow.
    """
    data = get_data()
    processed_data = process_data(data=data)
    return train_model(
        data=processed_data,
        logistic_regression=logistic_regression,
    )


if __name__ == "__main__":
    print(f"Running process_data() { process_data() }")
    print(f"Running training_workflow() { training_workflow() }")
