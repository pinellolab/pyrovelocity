from typing import Tuple

import torch


def mRNA(
    tau: torch.Tensor,
    u0: torch.Tensor,
    s0: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mRNA dynamics given the parameters and initial conditions.

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
        >>> mRNA(tau, u0, s0, alpha, beta, gamma)
        (tensor(1.1377), tensor(0.9269))
    """
    expu, exps = torch.exp(-beta * tau), torch.exp(-gamma * tau)

    # invalid values caused by below codes:
    # gamma equals beta will raise inf, inf * 0 leads to nan
    expus = (alpha - u0 * beta) * inv(gamma - beta) * (exps - expu)
    # solution 1: conditional zero filling
    # solution 1 issue:AutoDelta map_estimate of alpha,beta,gamma,switching will become nan, thus u_inf/s_inf/ut/st all lead to nan
    # expus = torch.where(torch.isclose(gamma, beta), expus.new_zeros(1), expus)

    ut = u0 * expu + alpha / beta * (1 - expu)
    st = (
        s0 * exps + alpha / gamma * (1 - exps) + expus
    )  # remove expus is the most stable, does it theoretically make sense?

    # solution 2: conditional analytical solution
    # solution 2 issue:AutoDelta map_estimate of alpha,beta,gamma,switching will become nan, thus u_inf/s_inf/ut/st all lead to nan
    # On the Mathematics of RNA Velocity I: Theoretical Analysis: Equation (2.12) when gamma == beta
    st2 = (
        s0 * expu + alpha / beta * (1 - expu) - (alpha - beta * u0) * tau * expu
    )
    ##st2 = s0 * expu + alpha / gamma * (1 - expu) - (alpha - gamma * u0) * tau * expu
    st = torch.where(torch.isclose(gamma, beta), st2, st)

    # solution 3: do not use AutoDelta and map_estimate? customize guide function?
    # use solution 3 with st2
    return ut, st


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
