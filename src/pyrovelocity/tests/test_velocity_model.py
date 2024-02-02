"""Tests for `pyrovelocity._velocity_model` module."""

import pytest
import torch
from pyro.distributions import Poisson
from pyro.nn import PyroModule

from pyrovelocity._velocity_model import LogNormalModel


def test_load__velocity_model():
    from pyrovelocity import _velocity_model

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
