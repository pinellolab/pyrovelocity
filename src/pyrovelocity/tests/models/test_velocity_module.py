"""Tests for `pyrovelocity._velocity_module` module."""


import pytest
import torch
from pyro.nn import PyroModule

from pyrovelocity.models._velocity_module import VelocityModule


def test_load__velocity_module():
    from pyrovelocity.models import _velocity_module

    print(_velocity_module.__file__)


class TestVelocityModule:
    @pytest.fixture
    def default_config(self):
        return {
            "num_cells": 100,
            "num_genes": 50,
            "model_type": "auto",
            "guide_type": "auto",
            "likelihood": "Poisson",
            "shared_time": True,
            "t_scale_on": False,
            "plate_size": 2,
            "latent_factor": "none",
            "latent_factor_operation": "selection",
            "latent_factor_size": 30,
            "inducing_point_size": 0,
            "include_prior": True,
            "use_gpu": "cpu",
            "num_aux_cells": 0,
            "only_cell_times": True,
            "decoder_on": False,
            "add_offset": False,
            "correct_library_size": True,
            "cell_specific_kinetics": None,
            "kinetics_num": None,
        }

    @pytest.fixture
    def velocity_module(self, default_config):
        return VelocityModule(**default_config)

    # def test_initialization(self, velocity_module, default_config):
    #     assert isinstance(velocity_module, PyroModule)
    #     assert velocity_module.num_cells == default_config["num_cells"]
    #     assert velocity_module.num_genes == default_config["num_genes"]
    #     assert velocity_module.guide_type == default_config["guide_type"]

    # @pytest.mark.parametrize("guide_type", ["auto", "auto_t0_constraint"])
    # def test_guide_type(self, default_config, guide_type):
    #     config = default_config.copy()
    #     config["guide_type"] = guide_type
    #     velocity_module = VelocityModule(**config)
    #     assert velocity_module.guide_type == guide_type

    # def test_forward(self, velocity_module):
    #     u_obs = torch.rand(100, 50)
    #     s_obs = torch.rand(100, 50)
    #     u_log_library = torch.rand(100, 1)
    #     s_log_library = torch.rand(100, 1)

    #     output = velocity_module(
    #         u_obs,
    #         s_obs,
    #         u_log_library,
    #         s_log_library,
    #         u_log_library,
    #         s_log_library,
    #         u_log_library,
    #         s_log_library,
    #         torch.arange(100),
    #     )

    #     assert isinstance(output, tuple)
    #     assert len(output) == 2
    #     assert all(isinstance(item, dict) for item in output)

    # def test_create_predictive(self, velocity_module):
    #     predictive = velocity_module.create_predictive()
    #     assert callable(predictive)

    # def test_get_fn_args_from_batch(self, velocity_module):
    #     batch = {
    #         "U": torch.rand(100, 50),
    #         "X": torch.rand(100, 50),
    #         "u_lib_size": torch.rand(100, 1),
    #         "s_lib_size": torch.rand(100, 1),
    #         "u_lib_size_mean": torch.rand(100, 1),
    #         "s_lib_size_mean": torch.rand(100, 1),
    #         "u_lib_size_scale": torch.rand(100, 1),
    #         "s_lib_size_scale": torch.rand(100, 1),
    #         "ind_x": torch.arange(100),
    #     }

    #     args, kwargs = velocity_module._get_fn_args_from_batch(batch)

    #     assert isinstance(args, tuple)
    #     assert len(args) == 10  # 9 from batch + 1 None for time_info
    #     assert isinstance(kwargs, dict)
    #     assert len(kwargs) == 0
