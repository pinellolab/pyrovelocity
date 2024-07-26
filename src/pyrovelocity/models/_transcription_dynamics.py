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
def atac_mrna_dynamics(
    tau: Tensor,
    c0: Tensor,
    u0: Tensor,
    s0: Tensor,
    k_c: Tensor,
    alpha_c: Tensor,
    alpha: Tensor,
    beta: Tensor,
    gamma: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes the ATAC and mRNA dynamics given temporal coordinate, parameter values, and
    initial conditions.

    `st_gamma_equals_beta` for the case where the gamma parameter is equal
    to the beta parameter is taken from Equation 2.12 of

    Args:
        tau (Tensor): Time points starting at last change in RNA transcription rate.
        c0 (Tensor): Initial value of c.
        u0 (Tensor): Initial value of u.
        s0 (Tensor): Initial value of s.
        k_c (Tensor): Chromatin state.
        alpha_c (Tensor): Rate of chromatin opening/closing.
        alpha (Tensor): Alpha parameter.
        beta (Tensor): Beta parameter.
        gamma (Tensor): Gamma parameter.

    Returns:
        Tuple[Tensor, Tensor]: Tuple containing the final values of c, u and s.

    Examples:
        >>> import torch
        >>> tau = torch.tensor(2.0)
        >>> c0 = torch.tensor(1.0)
        >>> u0 = torch.tensor(1.0)
        >>> s0 = torch.tensor(0.5)
        >>> alpha_c = torch.tensor(0.45)
        >>> alpha = torch.tensor(0.5)
        >>> beta = torch.tensor(0.4)
        >>> gamma = torch.tensor(0.3)
        >>> k_c = torch.tensor(1.0)
        >>> atac_mrna_dynamics(tau_c, tau, c0, u0, s0, k_c, alpha_c, alpha, beta, gamma)
        >>> import torch  
        >>> input = [torch.tensor([[ 0.,  0.],
                    [ 5.,  5.],
                    [35., 35.],
                    [15., 12.]]), torch.tensor([[0.0000, 0.0000],
                    [0.7769, 0.9502],
                    [0.7769, 0.9502],
                    [0.9985, 1.0000]]), torch.tensor([[0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [5.4451, 2.7188]]), torch.tensor([[0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [9.1791, 4.7921]]), torch.tensor([[0., 0.],
                    [1., 1.],
                    [1., 1.],
                    [0., 1.]]), torch.tensor([0.1000, 0.2000]), torch.tensor([[0.0000, 0.0000],
                    [0.5000, 0.3000],
                    [0.5000, 0.3000],
                    [0.5000, 0.0000]]), torch.tensor([0.0900, 0.1100]), torch.tensor([0.0500, 0.0600])]
            >>> tau_vec = input[0]
            >>> c0_vec = input[1]
            >>> u0_vec = input[2]
            >>> s0_vec = input[3]
            >>> k_c_vec = input[4]
            >>> alpha_c = input[5]
            >>> alpha_vec = input[6]
            >>> beta = input[7]
            >>> gamma = input[8]           
            >>> atac_mrna_dynamics(
            tau_vec, c0_vec, u0_vec, s0_vec, k_c_vec, alpha_c, alpha_vec, beta, gamma
            )
            (tensor([[0.0000, 0.0000],
            [0.8647, 0.9817],
            [0.9933, 1.0000],
            [0.2228, 1.0000]]), tensor([[0.0000, 0.0000],
            [1.6662, 1.1191],
            [5.1763, 2.6659],
            [3.2144, 0.7263]]), tensor([[0.0000, 0.0000],
            [2.2120, 1.2959],
            [8.2623, 4.3877],
            [4.3359, 2.3326]]))
    """

    A = torch.exp(-alpha_c * tau)
    B = torch.exp(-beta * tau)
    C = torch.exp(-gamma * tau)

    ct = c0 * A + k_c * (1 - A)
    ut = (
        u0 * B
        + alpha * k_c / beta * (1 - B)
        + (k_c - c0) * alpha / (beta - alpha_c) * (B - A)
    )
    st = s0 * C + alpha * k_c / gamma * (1 - C)
    +beta / (gamma - beta) * (
        (alpha * k_c) / beta - u0 - (k_c - c0) * alpha / (beta - alpha_c)
    ) * (C - B)
    +beta / (gamma - alpha_c) * (k_c - c0) * alpha / (beta - alpha_c) * (C - A)

    return ct, ut, st

@beartype
def get_initial_states(
    t0_state: Tensor,
    k_c_state: Tensor,
    alpha_c: Tensor,
    alpha_state: Tensor,
    beta: Tensor,
    gamma: Tensor,
    state: Tensor   
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes initial conditions of chromatin and mRNA in each cell.

    Args:
        t0_state (Tensor): The switch times of each gene (1 for each state).
        k_c_state (Tensor): The chromatin state in each state.
        alpha_c (Tensor): The chromatin opening and closing rate.
        alpha_state (Tensor): The transcription rate of each gene in each state.
        beta (Tensor): The splicing rate of each gene.
        gamma (Tensor): The degradation rate of each gene.
        state (Tensor): The state of each cell.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Tuple containing the initial conditions of 
        c, u and s for each cell.

    Examples:
        >>> import torch  
        >>> alpha_c = torch.tensor((0.1, 0.2))
        >>> beta = torch.tensor((0.09, 0.11))
        >>> gamma = torch.tensor((0.05, 0.06))
        >>> state = torch.tensor([[0, 0],[2, 2],[2, 2],[3, 3]]) 
        >>> k_c_state = torch.tensor([[0., 1., 1., 0., 0.], [0., 1., 1., 1., 0.]])
        >>> alpha_state = torch.tensor([[0.0000, 0.0000, 0.5000, 0.5000, 0.0000],[0.0000, 0.0000, 0.3000, 0.0000, 0.0000]])
        >>> t0_state = torch.tensor([[  0.,  10.,  25.,  75., 102.],[  0.,  10.,  25.,  78.,  95.]])   
        >>> get_initial_states(
        t0_state, k_c_state, alpha_c, alpha_state, beta, gamma, state
        )
        (torch.tensor([[0.0000, 0.0000],
         [0.7769, 0.9502],
         [0.7769, 0.9502],
         [0.9985, 1.0000]]),
 torch.tensor([[0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [5.4451, 2.7188]]),
 torch.tensor([[0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [9.1791, 4.7921]]))
    """

    n_genes = t0_state.shape[0]
    c0_state_list = [torch.zeros(n_genes),torch.zeros(n_genes)]
    u0_state_list = [torch.zeros(n_genes),torch.zeros(n_genes)]
    s0_state_list = [torch.zeros(n_genes),torch.zeros(n_genes)]
    dt_state = t0_state - torch.stack([torch.zeros((2)), torch.zeros((2)),
                                    t0_state[:,1], t0_state[:,2], t0_state[:,3]], dim = 1)                          # genes, states
    for i in range(1, 4):
        c0_i, u0_i, s0_i = atac_mrna_dynamics(
            dt_state[:, i+1], c0_state_list[-1], u0_state_list[-1], s0_state_list[-1], k_c_state[:, i],
            alpha_c, alpha_state[:, i], beta, gamma
        )
        c0_state_list += [c0_i] 
        u0_state_list += [u0_i]
        s0_state_list += [s0_i]
    
    c0_state = torch.stack(c0_state_list, dim = 1)
    u0_state = torch.stack(u0_state_list, dim = 1)
    s0_state = torch.stack(s0_state_list, dim = 1)
    
    c0_vec = c0_state[torch.arange(n_genes).unsqueeze(1), state.T].T                     # cells, genes
    u0_vec = u0_state[torch.arange(n_genes).unsqueeze(1), state.T].T                     # cells, genes
    s0_vec = s0_state[torch.arange(n_genes).unsqueeze(1), state.T].T                     # cells, genes

    return c0_vec, u0_vec, s0_vec

@beartype
def get_cell_parameters(
    t: Tensor,
    t0_1: Tensor,
    dt_1: Tensor,
    dt_2: Tensor,
    dt_3: Tensor,
    alpha: Tensor,
    alpha_off: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Gets the ODE parameters for each cell, by first assign each gene in each cell to a state 
    based on state switch times of a gene and then computes the transcription rate, chromatin state
    and time since last state switch(tau) for each gene in each cell.

    Args:
        t (Tensor): The time of each cell.
        t0_1 (Tensor): Start time for chromatin opening.
        dt_1 (Tensor): Time gap since chromatin opening for transcription start for each gene.
        dt_2 (Tensor): Time gap since transcription start for chromatin closing for each gene.
        dt_3 (Tensor): Time gap since transcription start for transcription stopping for each gene.
        alpha (Tensor): The transcription rate of each gene in the on state.
        alpha_off (Tensor): The transcription rate of each gene in the off state.

    Returns:
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple containing the state of each cell (state),
    the switch time of each state (t0_state), the chromatin opening state (k_c_state), the transcription rate in each cell
    (alpha_state) and cell-specific parameters for the chromatin state (k_c_vec) transcription rate (alpha_vec) and 
    time (tau_vec) since last state switch.

    Examples:
        >>> import torch  

        >>> n_cells = 4
        >>> t = torch.arange(0, 120, 30).reshape(n_cells, 1)
        >>> t0_1 = torch.tensor((10.0, 10.0))
        >>> dt_1 = torch.tensor((15.0, 15.0))
        >>> dt_2 = torch.tensor((77.0, 53.0))
        >>> dt_3 = torch.tensor((50.0, 70.0))
        >>> alpha = torch.tensor((0.5, 0.3))
        >>> alpha_off = torch.tensor(0.0)
        >>> get_cell_parameters(
            t, t0_1, dt_1, dt_2, dt_3, alpha, alpha_off
            )
        (tensor([[0, 0],
            [2, 2],
            [2, 2],
            [3, 3]]),tensor([[0., 1., 1., 0., 0.],
            [0., 1., 1., 1., 0.]]), tensor([[0.0000, 0.0000, 0.5000, 0.5000, 0.0000],
            [0.0000, 0.0000, 0.3000, 0.0000, 0.0000]]), tensor([[  0.,  10.,  25.,  75., 102.],
            [  0.,  10.,  25.,  78.,  95.]]), tensor([[0., 0.],
            [1., 1.],
            [1., 1.],
            [0., 1.]]), tensor([[0.0000, 0.0000],
            [0.5000, 0.3000],
            [0.5000, 0.3000],
            [0.5000, 0.0000]]), tensor([[ 0.,  0.],
            [ 5.,  5.],
            [35., 35.],
            [15., 12.]]))
    """

    # Assign each gene in each cell to a state:
    t0_2 = t0_1 + dt_1
    boolean = dt_2 >= dt_3 # True means chromatin starts closing, before transcription stops.
    t0_3 = torch.where(boolean, t0_2 + dt_3, t0_2 + dt_2)
    t0_4 = torch.where(~boolean, t0_2 + dt_3, t0_2 + dt_2)
    state = ((t0_1 <= t).int() + (t0_2 <= t).int() + (t0_3 <= t).int() + (t0_4 <= t).int())  # cells, genes
    n_genes = state.shape[1]
    
    t0_state = torch.stack([torch.zeros_like(t0_1), t0_1, t0_2, t0_3, t0_4], dim=1)  # genes, states
    t0_vec = t0_state[torch.arange(n_genes).unsqueeze(1), state.T].T  # cells, genes
    tau_vec = t - t0_vec  # cells, genes
    
    alpha_state = torch.stack([
        torch.ones_like(t0_1) * alpha_off,
        torch.ones_like(t0_1) * alpha_off,
        torch.ones_like(t0_1) * alpha,
        torch.where(boolean, torch.ones_like(t0_1) * alpha, torch.ones_like(t0_1) * alpha_off),
        torch.ones_like(t0_1) * alpha_off
    ], dim=1)  # genes, states

    k_c_state = torch.stack([
        torch.zeros_like(t0_1),
        torch.ones_like(t0_1),
        torch.ones_like(t0_1),
        torch.where(boolean, torch.zeros_like(t0_1), torch.ones_like(t0_1)),
        torch.zeros_like(t0_1)
    ], dim=1)  # genes, states

    alpha_vec = alpha_state[torch.arange(n_genes).unsqueeze(1), state.T].T  # cells, genes
    k_c_vec = k_c_state[torch.arange(n_genes).unsqueeze(1), state.T].T  # cells, genes
    
    return state, k_c_state, alpha_state, t0_state, k_c_vec, alpha_vec, tau_vec


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
