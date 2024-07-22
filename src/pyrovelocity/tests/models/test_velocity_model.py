"""Tests for _velocity_model.py"""

def test_MultiVelocityModelAuto():
        from pyrovelocity.models._velocity_model import MultiVelocityModelAuto
        
def test_MultiVelocityModelAuto_get_atac_rna():
        from pyrovelocity.models._velocity_model import MultiVelocityModelAuto
        import torch  

        n_cells = 4
        u_scale = torch.tensor(1.0)
        s_scale = torch.tensor(1.0)
        t = torch.arange(0, 120, 30).reshape(n_cells,1)                                           # cells, 1
        t0_1 = torch.tensor((10.0, 10.0))
        dt_1 = torch.tensor((15.0, 15.0))
        dt_2 = torch.tensor((77.0, 53.0))
        dt_3 = torch.tensor((50.0, 70.0))
        alpha_c = torch.tensor((0.1, 0.2))                          
        alpha = torch.tensor((0.5, 0.3))
        alpha_off = torch.tensor(0.0)
        beta = torch.tensor((0.09, 0.11))
        gamma = torch.tensor((0.05, 0.06))

        mod = MultiVelocityModelAuto(num_cells = n_cells, num_genes = 2)
        output = MultiVelocityModelAuto.get_atac_rna(
        mod, u_scale, s_scale, t, t0_1, dt_1, dt_2, dt_3, alpha_c, alpha, alpha_off, beta, gamma
        )

        correct_output = ((torch.tensor([[0.0000, 0.0000],
                [0.8647, 0.9817],
                [0.9933, 1.0000],
                [0.2228, 1.0000]]),
        torch.tensor([[0.0000, 0.0000],
                [1.6662, 1.1191],
                [5.1763, 2.6659],
                [3.2144, 0.7263]]),
        torch.tensor([[0.0000, 0.0000],
                [2.2120, 1.2959],
                [8.2623, 4.3877],
                [4.3359, 2.3326]])))
    
        for i in range(len(output)):
                assert torch.allclose(output[i], correct_output[i], atol=1e-3), f"Output at index {i} is incorrect"


