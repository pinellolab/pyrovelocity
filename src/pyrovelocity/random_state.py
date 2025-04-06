import os
import random

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Any, Dict


@beartype
def set_seed(
    seed: int,
    make_torch_deterministic: bool = True,
    make_cuda_deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Set random seeds for all libraries used in pyrovelocity.

    This function sets seeds for the following libraries:

    - python's random module
    - numpy
    - torch

    This is adapated from pyro, which also sets random, numpy, and pytorch seeds using
    the same approach via `pyro.set_rng_seed(seed)`.

    This function also configures pytorch to use deterministic algorithms where
    available.

    - [v2.6 torch reproducibility](https://pytorch.org/docs/2.6/notes/randomness.html#reproducibility)
    - [v2.6 torch.use_deterministic_algorithms](https://pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html)
    - [CUDA cublas results reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

    Args:
        seed: The seed value to use.
        make_torch_deterministic: Whether to use PyTorch's deterministic algorithms.
        make_cuda_deterministic: Whether to make CUDA operations deterministic.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        if make_cuda_deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if make_torch_deterministic:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)

    random.seed(seed)
    np.random.seed(seed)

    return get_state()


@beartype
def get_state() -> Dict[str, Any]:
    """
    Get the current random state of all libraries used in pyrovelocity.

    Returns:
        A dictionary containing the random states of python's random module,
        numpy, and pytorch.
    """
    state = {
        "torch": torch.get_rng_state(),
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = {
            i: torch.cuda.get_rng_state(i)
            for i in range(torch.cuda.device_count())
        }

    return state


@beartype
def set_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set the random state of all libraries used in pyrovelocity.

    Args:
        state: A dictionary containing the random states to set.
    """
    if "random" in state:
        random.setstate(state["random"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch" in state:
        torch.set_rng_state(state["torch"])

    if "torch_cuda" in state and torch.cuda.is_available():
        for device, rng_state in state["torch_cuda"].items():
            if device < torch.cuda.device_count():
                torch.cuda.set_rng_state(rng_state, device)

    return get_state()
