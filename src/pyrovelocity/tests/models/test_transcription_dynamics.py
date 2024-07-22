"""Tests for _transcription_dynamics_ functions."""

from pyrovelocity.models._transcription_dynamics import (
    atac_mrna_dynamics,
    get_cell_parameters,
    get_initial_states
)

def test_get_cell_parameters():
    import torch  

    n_cells = 4
    t = torch.arange(0, 120, 30).reshape(n_cells, 1)
    t0_1 = torch.tensor((10.0, 10.0))
    dt_1 = torch.tensor((15.0, 15.0))
    dt_2 = torch.tensor((77.0, 53.0))
    dt_3 = torch.tensor((50.0, 70.0))
    alpha = torch.tensor((0.5, 0.3))
    alpha_off = torch.tensor(0.0)
    
    output = get_cell_parameters(
        t, t0_1, dt_1, dt_2, dt_3, alpha, alpha_off
        )

    correct_output = (torch.tensor([[0, 0],
            [2, 2],
            [2, 2],
            [3, 3]]),torch.tensor([[0., 1., 1., 0., 0.],
            [0., 1., 1., 1., 0.]]), torch.tensor([[0.0000, 0.0000, 0.5000, 0.5000, 0.0000],
            [0.0000, 0.0000, 0.3000, 0.0000, 0.0000]]), torch.tensor([[  0.,  10.,  25.,  75., 102.],
            [  0.,  10.,  25.,  78.,  95.]]), torch.tensor([[0., 0.],
            [1., 1.],
            [1., 1.],
            [0., 1.]]), torch.tensor([[0.0000, 0.0000],
            [0.5000, 0.3000],
            [0.5000, 0.3000],
            [0.5000, 0.0000]]), torch.tensor([[ 0.,  0.],
            [ 5.,  5.],
            [35., 35.],
            [15., 12.]]))
    
    for i in range(len(output)):
            assert torch.allclose(output[i], correct_output[i], atol=1e-3), f"Output at index {i} is incorrect"
        
def test_get_initial_states():
    import torch  

    alpha_c = torch.tensor((0.1, 0.2))
    beta = torch.tensor((0.09, 0.11))
    gamma = torch.tensor((0.05, 0.06))
    state = torch.tensor([[0, 0],[2, 2],[2, 2],[3, 3]]) 
    k_c_state = torch.tensor([[0., 1., 1., 0., 0.], [0., 1., 1., 1., 0.]])
    alpha_state = torch.tensor([[0.0000, 0.0000, 0.5000, 0.5000, 0.0000],[0.0000, 0.0000, 0.3000, 0.0000, 0.0000]])
    t0_state = torch.tensor([[  0.,  10.,  25.,  75., 102.],[  0.,  10.,  25.,  78.,  95.]])
    
    output = get_initial_states(
    t0_state, k_c_state, alpha_c, alpha_state, beta, gamma, state
    )

    correct_output = (torch.tensor([[0.0000, 0.0000],
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
    
    for i in range(len(output)):
            assert torch.allclose(output[i], correct_output[i], atol=1e-3), f"Output at index {i} is incorrect"
            

def test_atac_mrna_dynamics():
    import torch  

    input = [torch.tensor([[ 0.,  0.],
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

    tau_vec = input[0]
    c0_vec = input[1]
    u0_vec = input[2]
    s0_vec = input[3]
    k_c_vec = input[4]
    alpha_c = input[5]
    alpha_vec = input[6]
    beta = input[7]
    gamma = input[8]
    
    output = atac_mrna_dynamics(
    tau_vec, c0_vec, u0_vec, s0_vec, k_c_vec, alpha_c, alpha_vec, beta, gamma
    )

    correct_output = (torch.tensor([[0.0000, 0.0000],
        [0.8647, 0.9817],
        [0.9933, 1.0000],
        [0.2228, 1.0000]]), torch.tensor([[0.0000, 0.0000],
        [1.6662, 1.1191],
        [5.1763, 2.6659],
        [3.2144, 0.7263]]), torch.tensor([[0.0000, 0.0000],
        [2.2120, 1.2959],
        [8.2623, 4.3877],
        [4.3359, 2.3326]]))
    
    for i in range(len(output)):
            assert torch.allclose(output[i], correct_output[i], atol=1e-3), f"Output at index {i} is incorrect"


