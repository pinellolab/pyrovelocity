import numpy as np
import pytest
from sparse import COO

from pyrovelocity.io.sparsity import (
    analyze_sparsification,
    calculate_density,
    check_sparsification_candidates,
    densify_arrays,
    print_sparsification_results,
    sparsify_arrays,
)


@pytest.fixture
def dense_array():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def sparse_array():
    return np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])


@pytest.fixture
def very_sparse_array():
    return np.array([[0, 0, 1], [0, 0, 0], [0, 2, 0]])


@pytest.fixture
def sample_dict_arrays(dense_array, sparse_array, very_sparse_array):
    return {
        "dense": dense_array,
        "sparse": sparse_array,
        "very_sparse": very_sparse_array,
    }


@pytest.fixture
def edge_cases_dict():
    return {
        "empty": np.array([]),
        "all_zeros": np.zeros((3, 3)),
        "single_element": np.array([1]),
        "single_zero": np.array([0]),
    }


@pytest.fixture
def large_sparse_array():
    arr = np.zeros((1000, 1000))
    arr[0, 0] = 1
    return arr


@pytest.mark.parametrize(
    "array,expected_density",
    [
        ("dense_array", 1.0),
        ("sparse_array", 1 / 3),
    ],
)
def test_calculate_density(array, expected_density, request):
    assert calculate_density(request.getfixturevalue(array)) == pytest.approx(
        expected_density
    )


def test_check_sparsification_candidates(sample_dict_arrays):
    candidates = check_sparsification_candidates(
        sample_dict_arrays, density_threshold=0.5
    )
    assert "dense" not in candidates
    assert "sparse" in candidates
    assert "very_sparse" in candidates


def test_print_sparsification_results(capsys, sample_dict_arrays):
    candidates = check_sparsification_candidates(
        sample_dict_arrays, density_threshold=0.5
    )
    print_sparsification_results(candidates)
    captured = capsys.readouterr()
    assert "Array 'sparse':" in captured.out
    assert "Array 'very_sparse':" in captured.out
    assert "Array 'dense':" not in captured.out


def test_analyze_sparsification(capsys, sample_dict_arrays):
    candidates = analyze_sparsification(
        sample_dict_arrays, density_threshold=0.5
    )
    assert "dense" not in candidates
    assert "sparse" in candidates
    assert "very_sparse" in candidates
    captured = capsys.readouterr()
    assert "Array 'sparse':" in captured.out
    assert "Array 'very_sparse':" in captured.out
    assert "Array 'dense':" not in captured.out


def test_sparsification_size_reduction(sample_dict_arrays):
    candidates = check_sparsification_candidates(
        sample_dict_arrays, density_threshold=0.5
    )
    for info in candidates.values():
        if info["density"] < 0.5:
            assert info["size_reduction"] >= 0.0


def test_edge_cases(edge_cases_dict):
    candidates = check_sparsification_candidates(
        edge_cases_dict, density_threshold=0.5
    )
    assert "empty" in candidates
    assert "all_zeros" in candidates
    assert "single_element" not in candidates
    assert "single_zero" in candidates


def test_large_array_performance(large_sparse_array):
    candidates = check_sparsification_candidates(
        {"large_sparse": large_sparse_array}, density_threshold=0.5
    )
    assert "large_sparse" in candidates
    assert candidates["large_sparse"]["density"] == pytest.approx(1 / 1_000_000)


@pytest.fixture
def mixed_dict_arrays(dense_array, sparse_array, very_sparse_array):
    return {
        "dense": dense_array,
        "sparse": sparse_array,
        "very_sparse": very_sparse_array,
        "already_coo": COO.from_numpy(np.array([[0, 1], [0, 0]])),
    }


def test_sparsify_arrays(mixed_dict_arrays):
    result = sparsify_arrays(mixed_dict_arrays, density_threshold=0.5)

    assert isinstance(result["dense"], np.ndarray)
    assert isinstance(result["sparse"], COO)
    assert isinstance(result["very_sparse"], COO)
    assert isinstance(result["already_coo"], COO)

    np.testing.assert_array_equal(result["dense"], mixed_dict_arrays["dense"])
    np.testing.assert_array_equal(
        result["sparse"].todense(), mixed_dict_arrays["sparse"]
    )
    np.testing.assert_array_equal(
        result["very_sparse"].todense(), mixed_dict_arrays["very_sparse"]
    )
    assert result["already_coo"] is mixed_dict_arrays["already_coo"]


@pytest.fixture
def coo_dict_arrays():
    return {
        "dense": np.array([[1, 2], [3, 4]]),
        "sparse": COO.from_numpy(np.array([[1, 0], [0, 2]])),
        "very_sparse": COO.from_numpy(np.array([[0, 1], [0, 0]])),
    }


def test_densify_arrays(coo_dict_arrays):
    result = densify_arrays(coo_dict_arrays)

    assert all(isinstance(arr, np.ndarray) for arr in result.values())
    np.testing.assert_array_equal(result["dense"], coo_dict_arrays["dense"])
    np.testing.assert_array_equal(
        result["sparse"], coo_dict_arrays["sparse"].todense()
    )
    np.testing.assert_array_equal(
        result["very_sparse"], coo_dict_arrays["very_sparse"].todense()
    )


def test_round_trip(sample_dict_arrays):
    sparsified = sparsify_arrays(sample_dict_arrays, density_threshold=0.5)
    densified = densify_arrays(sparsified)

    assert all(isinstance(arr, np.ndarray) for arr in densified.values())
    for key in sample_dict_arrays:
        np.testing.assert_array_equal(densified[key], sample_dict_arrays[key])
