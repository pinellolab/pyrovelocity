from typing import Tuple

import torch
from beartype import beartype
from torch import Tensor


@beartype
def mrna_dynamics(
    tau: Tensor,
    u0: Tensor,
    s0: Tensor,
    alpha: Tensor,
    beta: Tensor,
    gamma: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the mRNA dynamics given temporal coordinate, parameter values, and
    initial conditions.

    `st_gamma_equals_beta` for the case where the gamma parameter is equal
    to the beta parameter is taken from Equation 2.12 of

    > Li T, Shi J, Wu Y, Zhou P. On the mathematics of RNA velocity I:
    Theoretical analysis. CSIAM Transactions on Applied Mathematics. 2021;2:
    1–55. doi:[10.4208/csiam-am.so-2020-0001](https://doi.org/10.4208/csiam-am.so-2020-0001)

    Args:
        tau (Tensor): Time points.
        u0 (Tensor): Initial value of u.
        s0 (Tensor): Initial value of s.
        alpha (Tensor): Alpha parameter.
        beta (Tensor): Beta parameter.
        gamma (Tensor): Gamma parameter.

    Returns:
        Tuple[Tensor, Tensor]: Tuple containing the final values of u and s.

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
def inv(x: Tensor) -> Tensor:
    """
    Computes the element-wise reciprocal of a tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with element-wise reciprocal of x.

    Examples:
        >>> import torch
        >>> x = torch.tensor([2., 4., 0.5])
        >>> inv(x)
        tensor([0.5000, 0.2500, 2.0000])
    """
    return x.reciprocal()
