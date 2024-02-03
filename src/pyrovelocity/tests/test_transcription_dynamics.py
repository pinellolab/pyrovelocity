"""Tests for `pyrovelocity._transcription_dynamics` module."""

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models import mrna_dynamics


def test_load__transcription_dynamics():
    from pyrovelocity.models import _transcription_dynamics

    print(_transcription_dynamics.__file__)


def test_mRNA_correct_output():
    """Test the mRNA function with known inputs and outputs."""
    tau = torch.tensor(2.0)
    u0 = torch.tensor(1.0)
    s0 = torch.tensor(0.5)
    alpha = torch.tensor(0.5)
    beta = torch.tensor(0.4)
    gamma = torch.tensor(0.3)
    u_expected, s_expected = torch.tensor(1.1377), torch.tensor(0.9269)
    u, s = mrna_dynamics(tau, u0, s0, alpha, beta, gamma)
    assert torch.isclose(u, u_expected, atol=1e-4, rtol=1e-4)
    assert torch.isclose(s, s_expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("invalid_input", [None, "string", 5, 5.0, [1, 2, 3]])
def test_mRNA_invalid_type_input(invalid_input):
    """Test the mRNA function with various invalid input types."""
    with pytest.raises(BeartypeCallHintParamViolation):
        mrna_dynamics(
            invalid_input,
            invalid_input,
            invalid_input,
            invalid_input,
            invalid_input,
            invalid_input,
        )


def test_mRNA_special_case_gamma_equals_beta():
    """Test the mRNA function for the case where gamma is very close to beta."""
    tau = torch.tensor(2.0)
    u0 = torch.tensor(1.0)
    s0 = torch.tensor(0.5)
    alpha = torch.tensor(0.5)
    beta = torch.tensor(0.4)
    gamma = torch.tensor(0.4)  # Same as beta
    u, s = mrna_dynamics(tau, u0, s0, alpha, beta, gamma)
    assert u is not None
    assert s is not None


@pytest.mark.parametrize("value", [0, 1e-6, 1e6])
def test_mRNA_extreme_parameter_values(value):
    """Test the mRNA function with extreme parameter values."""
    tau = torch.tensor(value)
    u0 = torch.tensor(value)
    s0 = torch.tensor(value)
    alpha = torch.tensor(value)
    beta = torch.tensor(value)
    gamma = torch.tensor(value)
    u, s = mrna_dynamics(tau, u0, s0, alpha, beta, gamma)
    assert u is not None
    assert s is not None
    assert s is not None
