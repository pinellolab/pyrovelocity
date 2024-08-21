import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Tuple
from sparse import COO

__all__ = [
    "analyze_sparsification",
    "calculate_density",
    "check_sparsification_candidates",
    "densify_arrays",
    "print_sparsification_results",
    "sparsify_arrays",
]


@beartype
def calculate_density(arr: np.ndarray | COO) -> float:
    if isinstance(arr, COO):
        return arr.density

    total_elements = arr.size
    if total_elements == 0:
        return 0.0

    non_zero_count = np.count_nonzero(arr)
    return non_zero_count / total_elements


@beartype
def check_sparsification_candidates(
    data_dict: Dict[str, np.ndarray | COO],
    density_threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    candidates = {}
    for key, arr in data_dict.items():
        density = calculate_density(arr)
        if density <= density_threshold:
            sparse_arr = arr if isinstance(arr, COO) else COO.from_numpy(arr)
            original_size = (
                arr.nbytes if isinstance(arr, np.ndarray) else sparse_arr.nbytes
            )
            candidates[key] = {
                "shape": arr.shape,
                "density": density,
                "original_size": original_size,
                "sparse_size": sparse_arr.nbytes,
                "size_reduction": 1 - (sparse_arr.nbytes / original_size)
                if original_size > 0
                else 0,
            }
    return candidates


@beartype
def print_sparsification_results(candidates: Dict[str, Dict[str, Any]]) -> None:
    for key, info in candidates.items():
        print(f"Array '{key}':")
        print(f"  Shape: {info['shape']}")
        print(f"  Density: {info['density']:.2%}")
        print(f"  Original size: {info['original_size']} bytes")
        print(f"  Sparse size: {info['sparse_size']} bytes")
        print(f"  Size reduction: {info['size_reduction']} bytes")
        print()


@beartype
def analyze_sparsification(
    data_dict: Dict[str, np.ndarray],
    density_threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    candidates = check_sparsification_candidates(data_dict, density_threshold)
    print_sparsification_results(candidates)
    return candidates


@beartype
def sparsify_arrays(
    data_dict: Dict[str, np.ndarray | COO],
    density_threshold: float = 0.3,
) -> Dict[str, np.ndarray | COO]:
    candidates = analyze_sparsification(data_dict, density_threshold)
    result = {}
    for key, arr in data_dict.items():
        if key in candidates and not isinstance(arr, COO):
            result[key] = COO.from_numpy(arr)
        else:
            result[key] = arr
    return result


@beartype
def densify_arrays(
    data_dict: Dict[str, np.ndarray | COO]
) -> Dict[str, np.ndarray]:
    result = {}
    for key, arr in data_dict.items():
        if isinstance(arr, COO):
            result[key] = arr.todense()
        else:
            result[key] = arr
    return result
