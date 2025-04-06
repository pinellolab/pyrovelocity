"""Tests for `pyrovelocity.random_state` module."""

import random

import numpy as np
import torch

from pyrovelocity.random_state import get_state, set_seed, set_state


def test_set_seed_deterministic():
    """Test that set_seed produces deterministic behavior."""
    seed = 42

    set_seed(seed)
    state1 = get_state()

    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_seed(seed)
    state2 = get_state()

    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert state1["random"] == state2["random"]
    assert np.array_equal(state1["numpy"][1], state2["numpy"][1])
    assert torch.equal(state1["torch"], state2["torch"])

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


def test_set_seed_different_values():
    """Test that different seeds produce different random states."""
    set_seed(42)
    state1 = get_state()
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_seed(43)
    state2 = get_state()
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert state1["random"] != state2["random"]
    assert not np.array_equal(state1["numpy"][1], state2["numpy"][1])
    assert not torch.equal(state1["torch"], state2["torch"])

    assert r1 != r2
    assert n1 != n2
    assert t1 != t2


def test_get_state_contains_expected_keys():
    """Test that get_state returns a dict with expected keys."""
    set_seed(42)
    state = get_state()

    assert "random" in state
    assert "numpy" in state
    assert "torch" in state

    if torch.cuda.is_available():
        assert "torch_cuda" in state
        assert len(state["torch_cuda"]) == torch.cuda.device_count()


def test_set_state():
    """Test that set_state properly restores random states."""
    set_seed(42)
    state1 = get_state()
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_seed(43)
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1)

    set_state(state1)

    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


def test_set_state_returns_current_state():
    """Test that set_state returns the current state after setting."""
    set_seed(42)
    original_state = get_state()

    set_seed(43)

    returned_state = set_state(original_state)

    current_state = get_state()

    assert torch.equal(returned_state["torch"], current_state["torch"])
    assert returned_state["random"] == current_state["random"]

    np_ret = returned_state["numpy"]
    np_cur = current_state["numpy"]
    assert np_ret[0] == np_cur[0]
    assert np.array_equal(np_ret[1], np_cur[1])


def test_deterministic_algorithms():
    """Test that deterministic algorithms are being used."""
    seed = 42
    set_seed(seed, make_torch_deterministic=True)

    x = torch.randn(5, 5)
    result1 = x.mm(x)

    set_seed(seed, make_torch_deterministic=True)
    x = torch.randn(5, 5)
    result2 = x.mm(x)

    assert torch.allclose(result1, result2)


def test_set_state_with_partial_state():
    """Test that set_state handles partial state dictionaries correctly."""

    set_seed(42)
    full_state = get_state()

    random_val1 = random.random()
    numpy_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()

    set_seed(43)

    random_state = {"random": full_state["random"]}
    ret_state = set_state(random_state)

    assert "random" in ret_state
    assert "numpy" in ret_state
    assert "torch" in ret_state

    assert ret_state["random"] == full_state["random"]

    numpy_state = {"numpy": full_state["numpy"]}
    ret_state = set_state(numpy_state)

    assert np.array_equal(ret_state["numpy"][1], full_state["numpy"][1])

    torch_state = {"torch": full_state["torch"]}
    ret_state = set_state(torch_state)

    assert torch.equal(ret_state["torch"], full_state["torch"])

    set_seed(43)
    set_state(full_state)

    random_val2 = random.random()
    numpy_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()

    assert random_val1 == random_val2
    assert numpy_val1 == numpy_val2
    assert torch_val1 == torch_val2
