from typing import Tuple

import torch
from beartype import beartype


@beartype
def mrna_dynamics(
    tau: torch.Tensor,
    u0: torch.Tensor,
    s0: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mRNA dynamics given the parameters and initial conditions.

    `st_gamma_equals_beta` is taken from On the Mathematics of RNA Velocity I:
    Theoretical Analysis: Equation (2.12) when gamma == beta

    Args:
        tau (torch.Tensor): Time points.
        u0 (torch.Tensor): Initial value of u.
        s0 (torch.Tensor): Initial value of s.
        alpha (torch.Tensor): Alpha parameter.
        beta (torch.Tensor): Beta parameter.
        gamma (torch.Tensor): Gamma parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the final values of u and s.

    Examples:
        >>> import torch
        >>> tau = torch.tensor(2.0)
        >>> u0 = torch.tensor(1.0)
        >>> s0 = torch.tensor(0.5)
        >>> alpha = torch.tensor(0.5)
        >>> beta = torch.tensor(0.4)
        >>> gamma = torch.tensor(0.3)
        >>> mrna_dynamics(tau, u0, s0, alpha, beta, gamma)
        (tensor(1.1377), tensor(0.9269))
    """
    expu, exps = torch.exp(-beta * tau), torch.exp(-gamma * tau)

    expus = (alpha - u0 * beta) * inv(gamma - beta) * (exps - expu)

    ut = u0 * expu + alpha / beta * (1 - expu)
    st = s0 * exps + alpha / gamma * (1 - exps) + expus
    st_gamma_equals_beta = (
        s0 * expu + alpha / beta * (1 - expu) - (alpha - beta * u0) * tau * expu
    )
    st = torch.where(torch.isclose(gamma, beta), st_gamma_equals_beta, st)

    return ut, st


@beartype
def inv(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the element-wise reciprocal of a tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with element-wise reciprocal of x.

    Examples:
        >>> import torch
        >>> x = torch.tensor([2., 4., 0.5])
        >>> inv(x)
        tensor([0.5000, 0.2500, 2.0000])
    """
    return x.reciprocal()
