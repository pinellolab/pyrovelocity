from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)
from kfp.v2 import dsl
from kfp.v2.dsl import Dataset  # , Input, Model, Artifact
from kfp.v2.dsl import Output


def create_environment_log_component(
    base_image: str,
    display_name: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
):
    @dsl.component(
        base_image=base_image,
    )
    def environment_log(message: str, environment_info: Output[Dataset]):
        print(message)
        print("Active conda environment and installed packages:")
        import os
        import subprocess

        commands = [
            ["ls", "-alh"],
            ["pwd"],
            ["which", "python"],
            ["python", "--version"],
            ["mamba", "info"],
            ["mamba", "info", "--envs"],
            ["mamba", "list"],
            ["pip", "list"],
            ["pip", "freeze"],
            ["ls", "-alh", "/usr/local/nvidia/lib64"],
            ["/usr/local/nvidia/bin/nvidia-smi"],
        ]

        env_variables = ["NVIDIA_VISIBLE_DEVICES", "PATH", "LD_LIBRARY_PATH"]

        with open(environment_info.path, "a") as f:
            for command in commands:
                f.write(" ".join(command) + "\n")
                result = subprocess.run(command, capture_output=True, text=True)
                f.write(result.stdout + "\n" + result.stderr + "\n")
                print(result.stdout)
                print(result.stderr)

            for var in env_variables:
                f.write(f"{var}: {os.environ.get(var, 'Not Found')}\n")
                print(f"{var}: {os.environ.get(var, 'Not Found')}")

        import torch

        print(torch.__version__)

        if torch.cuda.is_available():
            print("A CUDA-enabled GPU is available.")
            for device in range(torch.cuda.device_count()):
                print(f"  Device: {device}")
                print(f"  Name: {torch.cuda.get_device_name(device)}")
                print(
                    f"  Compute capability: {torch.cuda.get_device_capability(device)}"
                )
                print(f"  Properties: {torch.cuda.get_device_properties(device)}")
        else:
            print("A CUDA-enabled GPU is not available.")

    return create_custom_training_job_from_component(
        environment_log,
        display_name=display_name,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )
