"""Tests for `pyrovelocity._velocity_model` module."""

import pytest
import torch
from pyro.distributions import Poisson
from pyro.nn import PyroModule

from pyrovelocity.models._velocity_model import LogNormalModel
from pyrovelocity.models._velocity_model import VelocityModelAuto


def test_load__velocity_model():
    from pyrovelocity.models import _velocity_model

    print(_velocity_model.__file__)


class TestLogNormalModel:
    @pytest.fixture
    def log_normal_model(self):
        return LogNormalModel(
            num_cells=3, num_genes=4, likelihood="Poisson", plate_size=2
        )

    def test_initialization(self, log_normal_model):
        """Test initialization of LogNormalModel"""
        assert isinstance(log_normal_model, PyroModule)
        assert log_normal_model.num_cells == 3
        assert log_normal_model.num_genes == 4
        assert log_normal_model.likelihood == "Poisson"
        assert log_normal_model.plate_size == 2

    def test_create_plates(self, log_normal_model):
        """Test create_plates method"""
        cell_plate, gene_plate = log_normal_model.create_plates()
        assert cell_plate.size == 3
        assert gene_plate.size == 4

    def test_get_likelihood(self, log_normal_model):
        """Test get_likelihood method"""
        ut = torch.rand(3, 4)
        st = torch.rand(3, 4)
        u_read_depth = torch.rand(3, 1)
        s_read_depth = torch.rand(3, 1)
        u_dist, s_dist = log_normal_model.get_likelihood(
            ut, st, u_read_depth=u_read_depth, s_read_depth=s_read_depth
        )
        assert isinstance(u_dist, Poisson)
        assert isinstance(s_dist, Poisson)

    def test_invalid_likelihood(self, log_normal_model):
        """Test get_likelihood method with invalid likelihood"""
        log_normal_model.likelihood = "Invalid"
        ut = torch.rand(3, 4)
        st = torch.rand(3, 4)
        with pytest.raises(NotImplementedError):
            log_normal_model.get_likelihood(ut, st)


class TestVelocityModelAuto:
    @pytest.fixture
    def velocity_model_auto(self):
        return VelocityModelAuto(
            num_cells=3,
            num_genes=4,
            likelihood="Poisson",
            shared_time=True,
            t_scale_on=False,
            plate_size=2,
            latent_factor="none",
            latent_factor_size=30,
            latent_factor_operation="selection",
            include_prior=False,
            num_aux_cells=100,
            only_cell_times=False,
            decoder_on=False,
            add_offset=False,
            correct_library_size=True,
            guide_type="velocity",
            cell_specific_kinetics=None,
            kinetics_num=None,
        )

    def test_initialization(self, velocity_model_auto):
        """Test initialization of VelocityModelAuto"""
        assert isinstance(velocity_model_auto, LogNormalModel)
        assert velocity_model_auto.num_cells == 3
        assert velocity_model_auto.num_genes == 4
        assert velocity_model_auto.likelihood == "Poisson"
        assert velocity_model_auto.shared_time is True
        assert velocity_model_auto.plate_size == 2

    def test_inherited_create_plates(self, velocity_model_auto):
        """Test inherited create_plates method from LogNormalModel"""
        cell_plate, gene_plate = velocity_model_auto.create_plates()
        assert cell_plate.size == 3
        assert gene_plate.size == 4

    def test_forward_method(self, velocity_model_auto):
        """Test the forward method"""
        u_obs = torch.rand(3, 4)
        s_obs = torch.rand(3, 4)
        u_log_library = torch.tensor([[3.7377], [4.0254], [2.7081]])
        s_log_library = torch.tensor([[3.6376], [3.9512], [2.3979]])
        u_log_library_loc = torch.tensor([[3.4904], [3.4904], [3.4904]])
        s_log_library_loc = torch.tensor([[3.3289], [3.3289], [3.3289]])
        u_log_library_scale = torch.tensor([[0.6926], [0.6926], [0.6926]])
        s_log_library_scale = torch.tensor([[0.8214], [0.8214], [0.8214]])
        ind_x = torch.tensor([2, 0, 1])

        u, s = velocity_model_auto.forward(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            u_log_library_loc=u_log_library_loc,
            s_log_library_loc=s_log_library_loc,
            u_log_library_scale=u_log_library_scale,
            s_log_library_scale=s_log_library_scale,
            ind_x=ind_x,
        )

        assert u.shape == (3, 4)
        assert s.shape == (3, 4)

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
