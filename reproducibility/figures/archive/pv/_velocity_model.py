from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.distributions import Bernoulli
from pyro.distributions import Beta
from pyro.distributions import Categorical
from pyro.distributions import Dirichlet
from pyro.distributions import Gamma
from pyro.distributions import LogNormal
from pyro.distributions import NegativeBinomial
from pyro.distributions import Normal
from pyro.distributions import Poisson
from pyro.distributions.constraints import positive
from pyro.nn import PyroModule
from pyro.nn import PyroParam
from pyro.nn import PyroSample
from pyro.ops.indexing import Vindex
from pyro.primitives import plate
from scvi.nn import Decoder
from scvi.nn import DecoderSCVI
from scvi.nn import FCLayers
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softplus

from .utils import mRNA
from .utils import tau_inv


class LogNormalModel(PyroModule):
    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        likelihood: str = "Poisson",
        plate_size: int = 2,
    ) -> None:
        assert num_cells > 0 and num_genes > 0
        super().__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.n_obs = None
        self.plate_size = plate_size
        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("one", torch.tensor(1.0))

        self.likelihood = likelihood
        if self.likelihood == "NB":
            self.u_px_r = PyroParam(
                torch.ones(self.num_genes), constraint=positive, event_dim=0
            )
            self.s_px_r = PyroParam(
                torch.ones(self.num_genes), constraint=positive, event_dim=0
            )

    def create_plates(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ) -> Tuple[plate, plate]:
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        return cell_plate, gene_plate

    def create_plate(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-1
        )
        return cell_plate

    @PyroSample
    def alpha(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def t_scale(self):
        return Normal(self.zero, self.one * 0.1)

    @PyroSample
    def beta(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one * 0.25)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def gamma(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def gene_offset(self):
        return Normal(self.zero, self.one)

    @PyroSample
    def u_scale(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one * 0.1)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def s_scale(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one * 0.1)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def u_inf(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one * 0.1)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def s_inf(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one * 0.1)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def dt_switching(self):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one)
            # return Normal(self.zero, self.one)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @PyroSample
    def latent_time(self):
        if self.shared_time:
            if self.plate_size == 2:
                return Normal(self.zero, self.one * 0.1).mask(
                    self.include_prior
                )  # with shared cell_time
            else:
                return (
                    Normal(self.zero, self.one * 0.1)
                    .expand((self.num_genes,))
                    .to_event(1)
                    .mask(self.include_prior)
                )  # with shared cell_time
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one).mask(
                self.include_prior
            )  # without shared cell_time
        return (
            LogNormal(self.zero, self.one).expand((self.num_genes,)).to_event(1)
        )  # .mask(False) # without shared cell_time

    @PyroSample
    def cell_time(self):
        if self.plate_size == 2:
            if self.shared_time:
                return Normal(
                    self.zero, self.one
                )  # .mask(self.include_prior) # mask=False generate the same estimation as initialization
                # return LogNormal(self.zero, self.one) #.mask(self.include_prior) # mask=False generate the same estimation as initialization
        return LogNormal(
            self.zero, self.one
        )  # .mask(False) # mask=False with LogNormal makes negative correlation

    @PyroSample
    def p_velocity(self):
        return Beta(self.one, self.one)

    @PyroSample
    def cell_codebook(self):
        return (
            Normal(self.zero, self.one / (10 * self.latent_factor_size))
            .expand((self.latent_factor_size, self.num_genes * 2))
            .to_event(2)
            .mask(self.include_prior)
        )

    @PyroSample
    def cell_code(self):
        return (
            Normal(self.zero, self.one)
            .expand((self.latent_factor_size,))
            .to_event(1)
            .mask(self.include_prior)
        )

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: Dict[str, torch.Tensor]
    ) -> Tuple[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            None,
            None,
        ],
        Dict[Any, Any],
    ]:
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        u_log_library_mean = tensor_dict["u_lib_size_mean"]
        s_log_library_mean = tensor_dict["s_lib_size_mean"]
        u_log_library_scale = tensor_dict["u_lib_size_scale"]
        s_log_library_scale = tensor_dict["s_lib_size_scale"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        if "pyro_cell_state" in tensor_dict:
            cell_state = tensor_dict["pyro_cell_state"]
        else:
            cell_state = None
        if "time_info" in tensor_dict:
            time_info = tensor_dict["time_info"]
        else:
            time_info = None
        return (
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_mean,
            s_log_library_mean,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        ), {}

    def get_likelihood(
        self,
        ut: torch.Tensor,
        st: torch.Tensor,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_scale: Optional[torch.Tensor] = None,
        s_scale: Optional[torch.Tensor] = None,
        u_read_depth: Optional[torch.Tensor] = None,
        s_read_depth: Optional[torch.Tensor] = None,
        u_cell_size_coef: None = None,
        ut_coef: None = None,
        s_cell_size_coef: None = None,
        st_coef: None = None,
    ) -> Tuple[Poisson, Poisson]:
        ##if not (self.likelihood in ['Normal', 'LogNormal']): # and u_scale is None and s_scale is None:
        ##    ut = pyro.sample("ut", Normal(ut, u_scale))
        ##    st = pyro.sample("st", Normal(st, s_scale))
        if self.likelihood == "NB":
            if self.correct_library_size:
                ut = relu(ut) + self.one * 1e-6
                st = relu(st) + self.one * 1e-6
                ut = pyro.deterministic("ut", ut, event_dim=0)
                st = pyro.deterministic("st", st, event_dim=0)
                if not (
                    self.guide_type in ["velocity_auto", "velocity_auto_depth"]
                ):  # time is learned from scaled u_obs/s_obs, no need to scale
                    ut = ut / torch.sum(ut, dim=-1, keepdim=True)
                    st = st / torch.sum(st, dim=-1, keepdim=True)
                ut = pyro.deterministic("ut_norm", ut, event_dim=0)
                st = pyro.deterministic("st_norm", st, event_dim=0)
                u_logits = ((ut + self.one * 1e-6) * u_read_depth).log() - (
                    self.u_px_r.exp() + self.one * 1e-6
                ).log()
                s_logits = ((st + self.one * 1e-6) * s_read_depth).log() - (
                    self.s_px_r.exp() + self.one * 1e-6
                ).log()
            else:
                ut = relu(ut)
                st = relu(st)
                ut = pyro.deterministic("ut", ut, event_dim=0)
                st = pyro.deterministic("st", st, event_dim=0)
                u_logits = (relu(ut) + self.one * 1e-6).log() - (
                    self.u_px_r.exp() + self.one * 1e-6
                ).log()
                s_logits = (relu(st) + self.one * 1e-6).log() - (
                    self.s_px_r.exp() + self.one * 1e-6
                ).log()
            u_dist = NegativeBinomial(
                total_count=self.u_px_r.exp(), logits=u_logits
            )
            s_dist = NegativeBinomial(
                total_count=self.s_px_r.exp(), logits=s_logits
            )
        elif self.likelihood == "Poisson":
            if self.correct_library_size:
                ut = relu(ut) + self.one * 1e-6
                st = relu(st) + self.one * 1e-6
                ut = pyro.deterministic("ut", ut, event_dim=0)
                st = pyro.deterministic("st", st, event_dim=0)
                ##if not (self.guide_type in ['velocity_auto', 'velocity_auto_depth']): # time is learned from scaled u_obs/s_obs, no need to scale
                if self.correct_library_size == "cell_size_regress":
                    ##ut = relu(self.one * u_read_depth + ut * ut_coef + u_cell_size_coef)+self.one*1e-6
                    ##st = relu(self.one * s_read_depth + st * st_coef + s_cell_size_coef)+self.one*1e-6
                    ##ut = relu(self.one * u_read_depth + ut * ut_coef)+self.one*1e-6
                    ##st = relu(self.one * s_read_depth + st * st_coef)+self.one*1e-6
                    ##ut = torch.exp(relu(self.one * u_read_depth + ut * ut_coef + u_cell_size_coef)+self.one*1e-6)
                    ##st = torch.exp(relu(self.one * s_read_depth + st * st_coef + s_cell_size_coef)+self.one*1e-6)
                    ##ut = torch.exp(self.one * u_read_depth + u_cell_size_coef)
                    ##st = torch.exp(self.one * s_read_depth + s_cell_size_coef)
                    ut_sum = torch.log(torch.sum(ut, dim=-1, keepdim=True))
                    st_sum = torch.log(torch.sum(st, dim=-1, keepdim=True))
                    ut = torch.log(ut)
                    st = torch.log(st)
                    ut = torch.exp(
                        ut_coef * ut
                        + u_cell_size_coef * (-ut_sum + u_read_depth)
                    )
                    st = torch.exp(
                        st_coef * st
                        + s_cell_size_coef * (-st_sum + s_read_depth)
                    )
                else:
                    ut = ut / torch.sum(ut, dim=-1, keepdim=True)
                    st = st / torch.sum(st, dim=-1, keepdim=True)
                ut = pyro.deterministic("ut_norm", ut, event_dim=0)
                st = pyro.deterministic("st_norm", st, event_dim=0)

                if self.correct_library_size != "cell_size_regress":
                    ut = (ut + self.one * 1e-6) * u_read_depth
                    st = (st + self.one * 1e-6) * s_read_depth
            else:
                ut = relu(ut)
                st = relu(st)
                ut = pyro.deterministic("ut", ut, event_dim=0)
                st = pyro.deterministic("st", st, event_dim=0)
                ut = ut + self.one * 1e-6
                st = st + self.one * 1e-6

            u_dist = Poisson(ut)
            s_dist = Poisson(st)
        elif self.likelihood == "Normal":
            if u_scale is not None and s_scale is not None:
                u_dist = Normal(
                    ut, u_scale
                )  # NOTE: add scale parameters significantly decrease ELBO
                s_dist = Normal(st, s_scale)
            else:
                u_dist = Normal(
                    ut, self.one * 0.1
                )  # NOTE: add scale parameters significantly decrease ELBO
                s_dist = Normal(st, self.one * 0.1)
        elif self.likelihood == "LogNormal":
            if u_scale is not None and s_scale is not None:
                u_dist = LogNormal(
                    (ut + self.one * 1e-6).log(), u_scale
                )  # NOTE: add scale parameters significantly decrease ELBO
                s_dist = LogNormal((st + self.one * 1e-6).log(), s_scale)
            else:
                u_dist = LogNormal(
                    ut, self.one * 0.1
                )  # NOTE: add scale parameters significantly decrease ELBO
                s_dist = LogNormal(st, self.one * 0.1)
        else:
            raise
        return u_dist, s_dist


class VelocityModel(LogNormalModel):
    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        likelihood: str = "Poisson",
        shared_time: bool = True,
        t_scale_on: bool = False,
        plate_size: int = 2,
        latent_factor: str = "none",
        latent_factor_size: int = 30,
        latent_factor_operation: str = "selection",
        include_prior: bool = False,
        num_aux_cells: int = 100,
        only_cell_times: bool = False,
        decoder_on: bool = False,
        add_offset: bool = False,
        correct_library_size: Union[bool, str] = True,
        guide_type: bool = "velocity",
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        **initial_values,
    ) -> None:
        assert num_cells > 0 and num_genes > 0
        super().__init__(num_cells, num_genes, likelihood, plate_size)
        # TODO set self.num_aux_cells in self.__init__, 10-200
        self.num_aux_cells = num_aux_cells
        self.num_aux_cells = num_aux_cells
        self.only_cell_times = only_cell_times
        self.guide_type = guide_type
        self.cell_specific_kinetics = cell_specific_kinetics
        self.k = kinetics_num

        self.mask = initial_values.get(
            "mask", torch.ones(self.num_cells, self.num_genes).bool()
        )
        for key in initial_values:
            self.register_buffer(f"{key}_init", initial_values[key])

        self.shared_time = shared_time
        self.t_scale_on = t_scale_on
        self.add_offset = add_offset
        self.plate_size = plate_size

        self.latent_factor = latent_factor
        self.latent_factor_size = latent_factor_size
        self.latent_factor_operation = latent_factor_operation
        self.include_prior = include_prior
        self.decoder_on = decoder_on
        self.correct_library_size = correct_library_size
        if self.decoder_on:
            self.decoder = Decoder(1, self.num_genes, n_layers=2)

    def model(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """max plate = 2 with cell and gene plate"""
        cell_plate, gene_plate = self.create_plates(
            u_obs, s_obs, u_log_library, s_log_library, ind_x
        )

        with gene_plate, poutine.mask(mask=self.include_prior):
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            switching = self.switching
            u_scale = self.u_scale  # if self.likelihood == 'Normal' else None
            s_scale = self.s_scale  # if self.likelihood == 'Normal' else None
            ##u_inf = self.u_inf
            ##s_inf = self.s_inf
            ##switching = tau_inv(u_inf, s_inf, self.zero, self.zero, alpha, beta, gamma)
            ##switching = pyro.sample("switching", Normal(switching, self.one*0.1))
            if self.t_scale_on and self.shared_time:
                t_scale = self.t_scale
            if self.latent_factor_operation == "selection":
                p_velocity = self.p_velocity

            # if self.latent_factor_operation == 'selection':
            #    velocity_genecellpair = pyro.sample("genecellpair_type", Bernoulli(self.one))
            if self.latent_factor == "linear":
                u_pcs_mean = pyro.sample(
                    "u_pcs_mean", Normal(self.zero, self.one)
                )
                s_pcs_mean = pyro.sample(
                    "s_pcs_mean", Normal(self.zero, self.one)
                )
            u_inf, s_inf = mRNA(
                switching, self.zero, self.zero, alpha, beta, gamma
            )
            # u_inf = pyro.sample("u_inf", Normal(u_inf, u_scale))
            # s_inf = pyro.sample("s_inf", Normal(s_inf, s_scale))

        if self.latent_factor == "linear":
            cell_codebook = self.cell_codebook
            with cell_plate, poutine.mask(mask=False):
                cell_code = self.cell_code
        if self.shared_time:
            with cell_plate:
                cell_time = self.cell_time

        if self.likelihood in ["NB", "Poisson"]:
            # with cell_plate:
            #    u_read_depth = pyro.sample('u_read_depth', dist.LogNormal(u_log_library, self.one))
            #    s_read_depth = pyro.sample('s_read_depth', dist.LogNormal(s_log_library, self.one))
            u_read_depth = None
            s_read_depth = None
        else:
            u_read_depth = None
            s_read_depth = None

        with cell_plate, gene_plate, poutine.mask(
            mask=pyro.subsample(self.mask.to(alpha.device), event_dim=0)
        ):
            t = self.latent_time
            if self.shared_time:
                if self.t_scale_on:
                    t = cell_time * t_scale + t
                else:
                    t = cell_time + t
            state = (
                pyro.sample("cell_gene_state", Bernoulli(logits=t - switching))
                == self.zero
            )
            u0_vec = torch.where(state, self.zero, u_inf)
            s0_vec = torch.where(state, self.zero, s_inf)
            alpha_vec = torch.where(state, alpha, self.zero)
            # tau = softplus(torch.where(state, t, t - switching))
            tau = relu(torch.where(state, t, t - switching))
            ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
            if self.latent_factor_operation == "selection":
                velocity_genecellpair = pyro.sample(
                    "genecellpair_type", Bernoulli(p_velocity)
                )
            if self.latent_factor == "linear":
                regressor_output = torch.einsum(
                    "abc,cd->ad", cell_code, cell_codebook.squeeze()
                )
                regressor_u = softplus(
                    regressor_output[..., : self.num_genes].squeeze()
                    + u_pcs_mean
                )
                regressor_s = softplus(
                    regressor_output[..., self.num_genes :].squeeze()
                    + s_pcs_mean
                )
            if self.latent_factor_operation == "selection":
                ut = torch.where(
                    velocity_genecellpair == self.one,
                    (ut * u_scale / s_scale),
                    softplus(regressor_u),
                )
                st = torch.where(
                    velocity_genecellpair == self.one, st, softplus(regressor_s)
                )
            elif self.latent_factor_operation == "sum":
                ut = ut * u_scale / s_scale + regressor_u
                st = st + regressor_s
            else:
                ut = ut * u_scale / s_scale
                st = st
            u_dist, s_dist = self.get_likelihood(
                ut,
                st,
                u_log_library,
                s_log_library,
                u_scale,
                s_scale,
                u_read_depth,
                s_read_depth,
            )
            u = pyro.sample("u", u_dist, obs=u_obs)
            s = pyro.sample("s", s_dist, obs=s_obs)
        return ut, st

    def model2(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """max plate = 1 with only cell plate"""
        cell_plate = self.create_plate(
            u_obs, s_obs, u_log_library, s_log_library, ind_x
        )
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        switching = self.switching
        u_scale = self.u_scale
        s_scale = self.s_scale
        with cell_plate:  # , poutine.mask(mask=pyro.subsample(self.mask.to(alpha.device), event_dim=1)):
            t = self.latent_time
            u_inf, s_inf = mRNA(
                switching, self.zero, self.zero, alpha, beta, gamma
            )
            state = (
                pyro.sample(
                    "cell_gene_state",
                    Bernoulli(logits=t - switching).to_event(1),
                )
                == self.zero
            )
            u0_vec = torch.where(state, self.zero, u_inf)
            s0_vec = torch.where(state, self.zero, s_inf)
            alpha_vec = torch.where(state, alpha, self.zero)
            tau = torch.where(state, t, t - switching).clamp(0.0)
            ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
            u_dist, s_dist = self.get_likelihood(
                ut, st, u_log_library, s_log_library, u_scale, s_scale
            )
            u = pyro.sample("u", u_dist.to_event(1), obs=u_obs)
            s = pyro.sample("s", s_dist.to_event(1), obs=s_obs)
        return ut, st

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.plate_size == 2:
            return self.model(u_obs, s_obs, u_log_library, s_log_library, ind_x)
        else:
            return self.model2(
                u_obs, s_obs, u_log_library, s_log_library, ind_x
            )


class AuxCellVelocityModel(VelocityModel):
    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """max plate = 2 with cell and gene plate"""
        cell_plate, gene_plate = self.create_plates(
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_loc,
            s_log_library_loc,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
        )
        with gene_plate, poutine.mask(mask=self.include_prior):
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            dt_switching = self.dt_switching
            u_scale = self.u_scale
            s_scale = self.s_scale
            # u_inf, s_inf = ode_mRNA(switching, self.zero, self.zero, alpha, beta, gamma)
            # if self.t_scale_on and self.shared_time:
            #    t_scale = self.t_scale
            #    gene_offset = self.gene_offset
            # else:
            #    t_scale = None
            #    gene_offset = None
            # u0 = pyro.sample("u0", Gamma(self.one, self.one*2).mask(True))
            # s0 = pyro.sample("s0", Gamma(self.one, self.one*2).mask(True))
            # u0 = torch.where(u0 >= u_inf, u_inf, u0)
            # s0 = torch.where(s0 >= s_inf, s_inf, s0)
            # t_scale = self.t_scale

            # u0 = s0 = self.zero
            u0 = pyro.sample("u_offset", LogNormal(self.zero, self.one))
            s0 = pyro.sample("s_offset", LogNormal(self.zero, self.one))
            t_scale = None

            if self.add_offset:
                gene_offset = self.gene_offset
            else:
                gene_offset = self.zero
            switching = dt_switching + gene_offset
            # u_inf, s_inf = mRNA(switching, self.zero, self.zero, alpha, beta, gamma)
            # u_inf, s_inf = mRNA(dt_switching, self.zero, self.zero, alpha, beta, gamma)
            u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)

            # u0 = torch.where(u0 > u_inf, u_inf, u0)
            # s0 = torch.where(s0 > s_inf, s_inf, s0)

            if self.latent_factor_operation == "selection":
                p_velocity = pyro.sample(
                    "p_velocity", Beta(self.one * 5, self.one)
                )
            else:
                p_velocity = None

            if self.latent_factor == "linear":
                u_pcs_mean = pyro.sample(
                    "u_pcs_mean", Normal(self.zero, self.one)
                )
                s_pcs_mean = pyro.sample(
                    "s_pcs_mean", Normal(self.zero, self.one)
                )
            else:
                u_pcs_mean, s_pcs_mean = None, None

        if self.likelihood in ["NB", "Poisson"]:
            # with cell_plate:
            #    u_read_depth = pyro.sample('u_read_depth', dist.LogNormal(u_log_library, self.one))
            #    s_read_depth = pyro.sample('s_read_depth', dist.LogNormal(s_log_library, self.one))
            u_read_depth = None
            s_read_depth = None
        else:
            u_read_depth = None
            s_read_depth = None

        # same as before, just refactored slightly
        # with cell_plate, gene_plate:
        #     cell_gene_mask = pyro.subsample(self.mask.to(alpha.device), event_dim=0)

        if self.latent_factor == "linear":
            cell_codebook = self.cell_codebook
        else:
            cell_codebook = None

        # with cell_plate, poutine.mask(mask=(u_obs > 0) & (s_obs > 0)):
        with cell_plate:  # , poutine.mask(mask=cell_gene_mask):
            # u_observed_total_dist = ...  # needs to be positive - maybe Poisson(num_genes * sequencing_depth)?
            # u_observed_total = pyro.sample("u_observed_total", u_observed_total_dist, obs=u_obs.sum(-2))
            # u_read_depth = pyro.sample('u_read_depth', LogNormal(u_log_library, self.one))
            # s_read_depth = pyro.sample('s_read_depth', LogNormal(s_log_library, self.one))
            with pyro.condition(data={"u": u_obs, "s": s_obs}):
                ut, st = self.generate_cell(
                    gene_plate,
                    alpha,
                    beta,
                    gamma,
                    switching,
                    u_inf,
                    s_inf,
                    u_scale,
                    s_scale,
                    u_log_library,
                    s_log_library,
                    u_pcs_mean=u_pcs_mean,
                    s_pcs_mean=s_pcs_mean,
                    cell_codebook=cell_codebook,
                    t_scale=t_scale,
                    gene_offset=gene_offset,
                    p_velocity=p_velocity,
                    decoder_on=self.decoder_on,
                    u_read_depth=u_read_depth,
                    s_read_depth=s_read_depth,
                    u0=u0,
                    s0=s0,
                )

        # new: add population of fake cells to introduce global correlation across cells
        if self.num_aux_cells > 0:
            with pyro.contrib.autoname.scope(prefix="aux"):
                # TODO set self.num_aux_cells in self.__init__, 10-200
                aux_cell_plate = pyro.plate(
                    "aux_cell_plate", self.num_aux_cells, dim=cell_plate.dim
                )
                with aux_cell_plate:
                    # aux_u_obs = pyro.param("aux_u_obs", lambda: self.aux_u_obs_init, constraint=positive, event_dim=0)  # TODO initialize
                    # aux_s_obs = pyro.param("aux_s_obs", lambda: self.aux_s_obs_init, constraint=positive, event_dim=0)  # TODO initialize
                    # TODO: change the parameter cells into fixed cells
                    # use a larger number of fixed cells
                    aux_u_obs = self.aux_u_obs_init
                    aux_s_obs = self.aux_s_obs_init
                    aux_u_log_library = torch.log(
                        aux_u_obs.sum(axis=-1)
                    )  # TODO define
                    aux_s_log_library = torch.log(
                        aux_s_obs.sum(axis=-1)
                    )  # TODO define
                    with pyro.condition(data={"u": aux_u_obs, "s": aux_s_obs}):
                        aux_ut, aux_st = self.generate_cell(
                            gene_plate,
                            alpha,
                            beta,
                            gamma,
                            switching,
                            u_inf,
                            s_inf,
                            u_scale,
                            s_scale,
                            aux_u_log_library,
                            aux_s_log_library,
                            u_pcs_mean=u_pcs_mean,
                            s_pcs_mean=s_pcs_mean,
                            cell_codebook=cell_codebook,
                            t_scale=t_scale,
                            gene_offset=gene_offset,
                            p_velocity=p_velocity,
                            decoder_on=self.decoder_on,
                            u_read_depth=u_read_depth,
                            s_read_depth=s_read_depth,
                            u0=u0,
                            s0=s0,
                        )
        # same as before: return only non-auxiliary cell predictions
        return ut, st

    def generate_cell(
        self,
        gene_plate,
        alpha,
        beta,
        gamma,
        switching,
        u_inf,
        s_inf,
        u_scale,
        s_scale,
        u_log_library,
        s_log_library,
        u_pcs_mean=None,
        s_pcs_mean=None,
        cell_codebook=None,
        t_scale=None,
        gene_offset=None,
        p_velocity=None,
        decoder_on=False,
        u_read_depth=None,
        s_read_depth=None,
        u0=None,
        s0=None,
    ):
        if self.decoder_on:
            pyro.module("time_decoder", self.decoder)
        if self.latent_factor == "linear":
            cell_code = pyro.sample(
                "cell_code",
                Normal(self.zero, self.one)
                .expand((self.latent_factor_size,))
                .to_event(1)
                .mask(self.include_prior),
            )
        else:
            cell_code = None

        if self.shared_time:
            ##cell_time = pyro.sample("cell_time", LogNormal(self.zero, self.one).mask(False)) # mask=False works for cpm and raw read count, count needs steady-state initialization
            cell_time = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )  # mask=False works for cpm and raw read count, count needs steady-state initialization
        ##assert cell_time.shape == (256, 1)

        with gene_plate:
            if self.latent_factor_operation == "selection":
                cellgene_type = pyro.sample(
                    "cellgene_type", Bernoulli(p_velocity)
                )
            else:
                cellgene_type = None

            if not self.only_cell_times:
                if self.shared_time:
                    t = pyro.sample(
                        "latent_time",
                        Normal(self.zero, self.one).mask(self.include_prior),
                    )
                    if self.t_scale_on and t_scale is not None:
                        t = cell_time * t_scale + t + gene_offset
                    else:
                        t = cell_time + t  # + gene_offset
                else:
                    t = pyro.sample(
                        "latent_time",
                        LogNormal(self.zero, self.one).mask(self.include_prior),
                    )
            else:
                if self.decoder_on:
                    t, _ = self.decoder(cell_time)
                else:
                    t = cell_time

            ##state = pyro.sample("cell_gene_state", Bernoulli(logits=t-switching)) == self.zero
            state = (
                pyro.sample(
                    "cell_gene_state",
                    Bernoulli(logits=t - switching),
                    infer={"enumerate": "sequential"},
                )
                == self.zero
            )
            u0_vec = torch.where(state, u0, u_inf)
            s0_vec = torch.where(state, s0, s_inf)
            alpha_vec = torch.where(state, alpha, self.zero)

            tau = softplus(torch.where(state, t - gene_offset, t - switching))
            # tau = relu(torch.where(state, t-gene_offset, t - switching))

            ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
            # ut, st = ode_mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
            if self.latent_factor == "linear":
                regressor_output = torch.einsum(
                    "abc,cd->ad", cell_code, cell_codebook.squeeze()
                )
                regressor_u = softplus(
                    regressor_output[..., : self.num_genes].squeeze()
                    + u_pcs_mean
                )
                regressor_s = softplus(
                    regressor_output[..., self.num_genes :].squeeze()
                    + s_pcs_mean
                )
            if self.latent_factor_operation == "selection":
                ut = torch.where(
                    cellgene_type == self.one,
                    ut * u_scale / s_scale,
                    softplus(regressor_u),
                )
                st = torch.where(
                    cellgene_type == self.one, st, softplus(regressor_s)
                )
            elif self.latent_factor_operation == "sum":
                ut = ut * u_scale / s_scale + regressor_u
                st = st + regressor_s
            else:
                ut = ut * u_scale / s_scale
                st = st
            u_dist, s_dist = self.get_likelihood(
                ut,
                st,
                u_log_library,
                s_log_library,
                u_scale,
                s_scale,
                u_read_depth=u_read_depth,
                s_read_depth=s_read_depth,
            )
            u = pyro.sample("u", u_dist)
            s = pyro.sample("s", s_dist)
        return ut, st


class TimeEncoder(nn.Module):
    """adapt https://docs.scvi-tools.org/en/0.9.1/_modules/scvi/nn/_base_components.html#Encoder
    for shap DeepExplainer usage
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        # self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.act = nn.Softplus(beta=1)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        # Parameters for latent distribution
        # x = torch.hstack([x, u_log_library, s_log_library]) # insert library sizes
        q = self.encoder(x, *cat_list)
        q_m = self.act(
            self.mean_encoder(q)
        )  # really big cell_time, this also generates reverse order with poisson
        return q_m


class TimeEncoder2(nn.Module):
    """adapt https://docs.scvi-tools.org/en/0.9.1/_modules/scvi/nn/_base_components.html#Encoder
    for shap DeepExplainer usage
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-6,
        last_layer_activation=nn.Softplus(beta=1),
        **kwargs,
    ):
        super().__init__()
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.act = last_layer_activation
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        q = self.encoder(x, *cat_list)
        q_m = self.act(self.mean_encoder(q))
        q_v = self.act(self.var_encoder(q)) + self.var_eps
        return q_m, q_v


class BlockedKineticsModel(AuxCellVelocityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.time_encoder = TimeEncoder(
        self.time_encoder = TimeEncoder2(
            self.num_genes,
            n_output=1,
            dropout_rate=0.5,
            activation_fn=nn.ELU,
            n_layers=3,
            var_eps=1e-6,
        )
        if self.correct_library_size:
            self.u_lib_encoder = TimeEncoder2(self.num_genes + 1, 1, n_layers=2)
            self.s_lib_encoder = TimeEncoder2(self.num_genes + 1, 1, n_layers=2)

    def get_time(
        self,
        u_scale,
        s_scale,
        alpha,
        beta,
        gamma,
        u_obs,
        s_obs,
        u0,
        s0,
        t0,
        dt_switching,
        u_inf,
        s_inf,
        u_offset=None,
        s_offset=None,
    ):
        scale = u_scale / s_scale
        if u_offset is None:
            u_ = u_obs
            s_ = s_obs
        else:
            u_ = relu(u_obs - u_offset)
            s_ = relu(s_obs - s_offset)

        u_ = u_ / scale
        tau = tau_inv(u_, s_, u0, s0, alpha, beta, gamma)
        ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
        tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
        ut_, st_ = mRNA(tau_, u_inf, s_inf, self.zero, beta, gamma)
        std_u = u_scale / scale
        state_on = ((ut - u_) / std_u) ** 2 + ((st - s_) / s_scale) ** 2
        state_off = ((ut_ - u_) / std_u) ** 2 + ((st_ - s_) / s_scale) ** 2
        state_zero = ((ut - u0) / std_u) ** 2 + ((st - s0) / s_scale) ** 2
        state_inf = ((ut_ - u_inf) / std_u) ** 2 + (
            (st_ - s_inf) / s_scale
        ) ** 2
        cell_gene_state_logits = torch.stack(
            [state_on, state_zero, state_off, state_inf], dim=-1
        ).argmin(-1)
        state = (cell_gene_state_logits > 1) == self.zero
        t = torch.where(state, tau + t0, tau_ + dt_switching + t0)
        if self.guide_type == "auto":
            cell_time_loc, cell_time_scale = self.time_encoder(t)
            t = pyro.deterministic("cell_time", cell_time_loc, event_dim=0)
        elif self.guide_type == "velocity_auto":
            cell_time_loc, cell_time_scale = self.time_encoder(t)
            # t = pyro.sample("cell_time", Delta(cell_time_loc, event_dim=0))
            t = pyro.sample(
                "cell_time", LogNormal(cell_time_loc, cell_time_scale)
            )
        else:
            raise NotImplementedError
        return t

    def get_rna(
        self,
        u_scale,
        s_scale,
        alpha,
        beta,
        gamma,
        t,
        u0,
        s0,
        t0,
        switching,
        u_inf,
        s_inf,
    ):
        state = (
            pyro.sample(
                "cell_gene_state",
                Bernoulli(logits=t - switching),
                infer={"enumerate": "sequential"},
            )
            == self.zero
        )
        u0_vec = torch.where(state, u0, u_inf)
        s0_vec = torch.where(state, s0, s_inf)
        alpha_vec = torch.where(state, alpha, self.zero)
        tau = softplus(torch.where(state, t - t0, t - switching))
        ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
        ut = ut * u_scale / s_scale
        return ut, st

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        if self.guide_type == "auto":
            pyro.module("time_encoder", self.time_encoder)
        cell_plate, gene_plate = self.create_plates(
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_loc,
            s_log_library_loc,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
        )

        with gene_plate, poutine.mask(mask=self.include_prior):
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            dt_switching = self.dt_switching
            t0 = pyro.sample("t0", Normal(self.zero, self.one))
            u0 = s0 = self.zero
            if self.add_offset:
                u_offset = pyro.sample(
                    "u_offset", LogNormal(self.zero, self.one)
                )  # work with deeper network >=3 if start at zero
                s_offset = pyro.sample(
                    "s_offset", LogNormal(self.zero, self.one)
                )

            u_scale = self.u_scale
            s_scale = self.s_scale
            u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
            u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
            s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
            switching = pyro.deterministic(
                "switching", dt_switching + t0, event_dim=0
            )

        with cell_plate:
            if self.guide_type == "auto":
                t = self.get_time(
                    u_scale,
                    s_scale,
                    alpha,
                    beta,
                    gamma,
                    u_obs,
                    s_obs,
                    u0,
                    s0,
                    t0,
                    dt_switching,
                    u_inf,
                    s_inf,
                )
            elif self.guide_type == "velocity_auto":
                t = pyro.sample(
                    "cell_time",
                    LogNormal(self.zero, self.one).mask(self.include_prior),
                )
            else:
                raise NotImplementedError

            if self.correct_library_size:
                u_read_depth = pyro.sample(
                    "u_read_depth",
                    LogNormal(u_log_library, u_log_library_scale),
                )
                s_read_depth = pyro.sample(
                    "s_read_depth",
                    LogNormal(s_log_library, s_log_library_scale),
                )
            else:
                u_read_depth = s_read_depth = None
            with gene_plate:
                ####pyro.sample("time_constraint", Bernoulli(logits=t-t0), obs=self.one) # this constraint not work with NN guide for time
                ut, st = self.get_rna(
                    u_scale,
                    s_scale,
                    alpha,
                    beta,
                    gamma,
                    t,
                    u0,
                    s0,
                    t0,
                    switching,
                    u_inf,
                    s_inf,
                )

                if self.add_offset:
                    ut = ut + u_offset
                    st = st + s_offset

                u_dist, s_dist = self.get_likelihood(
                    ut,
                    st,
                    u_log_library,
                    s_log_library,
                    u_scale,
                    s_scale,
                    u_read_depth=u_read_depth,
                    s_read_depth=s_read_depth,
                )
                u = pyro.sample("u", u_dist, obs=u_obs)
                s = pyro.sample("s", s_dist, obs=s_obs)
        return u, s


class MultiKineticsModel(VelocityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.gamma_celltime_encoder = Encoder(1, n_output=1, n_layers=2)
        # self.beta_celltime_encoder = Encoder(1, n_output=1, n_layers=2)
        # self.alpha_celltime_encoder = Encoder(1, n_output=1, n_layers=2)
        print(self.k)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        u_log_library_mean = tensor_dict["u_lib_size_mean"]
        s_log_library_mean = tensor_dict["s_lib_size_mean"]
        u_log_library_scale = tensor_dict["u_lib_size_scale"]
        s_log_library_scale = tensor_dict["s_lib_size_scale"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        if "pyro_cell_state" in tensor_dict:
            cell_state = tensor_dict["pyro_cell_state"]
        else:
            cell_state = None
        if "time_info" in tensor_dict:
            time_info = tensor_dict["time_info"]
        else:
            time_info = None
        return (
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_mean,
            s_log_library_mean,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        ), {}

    def create_plates(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        kinetics_plate = pyro.plate("kinetics", self.k, dim=-3)
        return cell_plate, gene_plate, kinetics_plate

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pyro.module("gamma_celltime_encoder", self.gamma_celltime_encoder)
        # pyro.module("beta_celltime_encoder", self.beta_celltime_encoder)
        # pyro.module("alpha_celltime_encoder", self.alpha_celltime_encoder)
        # pyro.module("switching_celltime_encoder", self.switching_celltime_encoder)
        cell_plate, gene_plate, kinetics_plate = self.create_plates(
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_loc,
            s_log_library_loc,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        )

        with kinetics_plate, gene_plate, poutine.mask(mask=self.include_prior):
            alpha_k = pyro.sample("alpha", LogNormal(self.zero, self.one))
            beta_k = pyro.sample("beta", LogNormal(self.zero, self.one))
            gamma_k = pyro.sample("gamma", LogNormal(self.zero, self.one))
            dt_switching_k = pyro.sample(
                "dt_switching", LogNormal(self.zero, self.one)
            )
            assert alpha_k.shape == (kinetics_plate.size, 1, gene_plate.size)
            # t0_k = pyro.sample("t0", Normal(self.zero, self.one))
            # t0_k = pyro.sample("t0", LogNormal(self.zero, self.one))
            t0_k = pyro.sample("t0", Gamma(self.one, self.one))

            if self.add_offset:
                u_offset_k = pyro.sample(
                    "u_offset", LogNormal(self.zero, self.one)
                )
                s_offset_k = pyro.sample(
                    "s_offset", LogNormal(self.zero, self.one)
                )
            else:
                u_offset_k = s_offset_k = self.zero

            ###u_offset_k = s_offset_k = self.zero
            # u_inf, s_inf = mRNA(dt_switching, self.zero, self.zero, alpha, beta, gamma)
            # switching = t0_k + dt_switching
            # dt_switching = self.dt_switching
            # u_scale = self.u_scale
            # s_scale = self.s_scale
        # with gene_plate, poutine.mask(mask=self.include_prior):
        #    u_scale = self.u_scale
        #    s_scale = self.s_scale
        #    t0 = pyro.sample("t0", Normal(self.zero, self.one))
        #    u0 = u_offset = pyro.sample("u_offset", LogNormal(self.zero, self.one))
        #    s0 = s_offset = pyro.sample("s_offset", LogNormal(self.zero, self.one))
        # alpha = alpha_k[cluster_ind].squeeze()
        # beta = beta_k[cluster_ind].squeeze()
        # gamma = gamma_k[cluster_ind].squeeze()
        # t0 = t0_k[cluster_ind].squeeze()
        # u0 = u_offset = u_offset_k[cluster_ind].squeeze()
        # s0 = s_offset = s_offset_k[cluster_ind].squeeze()
        # u_offset = u_offset_k[cluster_ind].squeeze()
        # s_offset = s_offset_k[cluster_ind].squeeze()
        # u_scale = u_scale[cluster_ind].squeeze()
        # s_scale = s_scale[cluster_ind].squeeze()
        # dt_switching = dt_switching[cluster_ind].squeeze()

        u_inf_k, s_inf_k = mRNA(
            dt_switching_k, u_offset_k, s_offset_k, alpha_k, beta_k, gamma_k
        )
        switching_k = t0_k + dt_switching_k
        assert switching_k.shape == (kinetics_plate.size, 1, gene_plate.size)
        assert u_inf_k.shape == (kinetics_plate.size, 1, gene_plate.size)

        with cell_plate:
            cell_time = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )
            if self.correct_library_size and (self.likelihood != "Normal"):
                u_read_depth = pyro.sample(
                    "u_read_depth",
                    LogNormal(u_log_library, u_log_library_scale),
                )
                s_read_depth = pyro.sample(
                    "s_read_depth",
                    LogNormal(s_log_library, s_log_library_scale),
                )
                if self.correct_library_size == "cell_size_regress":
                    # cell-wise coef per cell
                    u_cell_size_coef = pyro.sample(
                        "u_cell_size_coef", Normal(self.zero, self.one)
                    )
                    ut_coef = pyro.sample(
                        "ut_coef", Normal(self.zero, self.one)
                    )

                    s_cell_size_coef = pyro.sample(
                        "s_cell_size_coef", Normal(self.zero, self.one)
                    )
                    st_coef = pyro.sample(
                        "st_coef", Normal(self.zero, self.one)
                    )
                else:
                    u_cell_size_coef = (
                        ut_coef
                    ) = s_cell_size_coef = st_coef = None
            else:
                u_read_depth = s_read_depth = None
                u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None

        # alpha_encode_cell_time, _, _ = self.alpha_celltime_encoder(cell_time)
        # beta_encode_cell_time, _, _ = self.beta_celltime_encoder(cell_time)
        # gamma_encode_cell_time, _, _ = self.gamma_celltime_encoder(cell_time)
        # switching_encode_cell_time = self.switching_celltime_encoder(cell_time)
        ##with cell_plate, gene_plate, poutine.mask(mask=(u_obs > 0) & (s_obs > 0)):
        logits_k = cell_time - switching_k
        assert logits_k.shape == (
            kinetics_plate.size,
            cell_plate.subsample_size,
            gene_plate.size,
        )

        with kinetics_plate, cell_plate, gene_plate:
            if (
                self.guide_type == "auto_t0_constraint"
                or self.guide_type == "velocity_auto_t0_constraint"
            ):
                pyro.sample(
                    "time_constraint",
                    Bernoulli(logits=cell_time - t0_k),
                    obs=self.one,
                )

        with kinetics_plate, cell_plate, gene_plate:
            state_k = (
                pyro.sample(
                    "cell_gene_state",
                    Bernoulli(logits=logits_k),
                    infer={"enumerate": "sequential"},
                )
                == self.zero
            )
            assert state_k.shape == (
                kinetics_plate.size,
                cell_plate.subsample_size,
                gene_plate.size,
            )
            u0_k = torch.where(state_k, u_offset_k, u_inf_k)
            s0_k = torch.where(state_k, s_offset_k, s_inf_k)
            alpha_k = torch.where(state_k, alpha_k, self.zero)
            tau_k = softplus(torch.where(state_k, cell_time - t0_k, logits_k))
            ut_k, st_k = mRNA(tau_k, u0_k, s0_k, alpha_k, beta_k, gamma_k)

            if self.correct_library_size and (self.likelihood != "Normal"):
                # mse = (ut_k - u_obs/u_read_depth)**2+(st_k - s_obs/s_read_depth)**2
                mse = (ut_k - u_obs) ** 2 + (st_k - s_obs) ** 2
            else:
                mse = (ut_k - u_obs) ** 2 + (st_k - s_obs) ** 2
            assert mse.shape == (
                kinetics_plate.size,
                cell_plate.subsample_size,
                gene_plate.size,
            )

        with cell_plate:
            mse = mse.permute(1, 2, 0)
            assert mse.shape == (
                cell_plate.subsample_size,
                gene_plate.size,
                kinetics_plate.size,
            )
            mse = mse.mean(axis=-2, keepdims=True)

            assert mse.shape == (
                cell_plate.subsample_size,
                1,
                kinetics_plate.size,
            )
            kinetics_lineage = pyro.sample(
                "mse",
                Categorical(logits=mse),
                infer={"enumerate": "sequential"},
            )
            # infer={'enumerate': 'parallel'})
            assert kinetics_lineage.shape == (cell_plate.subsample_size, 1)
            # kinetics_lineage = kinetics_lineage.view(-1, cell_plate.subsample_size, gene_plate.size)
            # kinetics_lineage = kinetics_lineage.view(1, cell_plate.subsample_size, 1)

            with gene_plate:
                # ut = ut_k.gather(0, kinetics_lineage).squeeze()
                # st = st_k.gather(0, kinetics_lineage).squeeze()
                # print(ut.shape)
                ut = torch.where(kinetics_lineage == 0, ut_k[0], ut_k[1])
                st = torch.where(kinetics_lineage == 0, st_k[0], st_k[1])

                assert ut.shape == (cell_plate.subsample_size, gene_plate.size)
                assert st.shape == (cell_plate.subsample_size, gene_plate.size)

                u_dist, s_dist = self.get_likelihood(
                    ut,
                    st,
                    u_log_library,
                    s_log_library,
                    u_scale=None,
                    s_scale=None,
                    u_read_depth=u_read_depth,
                    s_read_depth=s_read_depth,
                )
                u = pyro.sample("u", u_dist, obs=u_obs)
                s = pyro.sample("s", s_dist, obs=s_obs)
            ##pyro.sample("time_constraint", Bernoulli(logits=cell_time-t0), obs=self.one)

        # with cell_plate, gene_plate:
        #    #alpha = alpha * softplus(alpha_encode_cell_time)
        #    #beta = beta * softplus(beta_encode_cell_time)
        #    #gamma = gamma * softplus(gamma_encode_cell_time)
        #    #u_inf, s_inf = mRNA(dt_switching, self.zero, self.zero, alpha, beta, gamma)
        #    #switching = t0 + dt_switching
        #    #switching = relu(switching * switching_encode_cell_time)
        #    state = pyro.sample("cell_gene_state", Bernoulli(logits=cell_time-switching),
        #                        infer={'enumerate': 'sequential'}) == self.zero
        #    u0 = torch.where(state, u0, u_inf)
        #    s0 = torch.where(state, s0, s_inf)
        #    alpha = torch.where(state, alpha, self.zero)
        #    tau = relu(torch.where(state, cell_time - t0, cell_time-switching))
        #    #tau = relu(torch.where(state, switching - cell_time, cell_time-switching))
        #    #tau = relu(cell_time - t0)
        #    ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
        #    #ut, st = ode_mRNA(tau, u0, s0, alpha, beta, gamma)
        #    #ut = (ut + u_offset) * u_scale / s_scale
        #    #st = st + s_offset
        #    ut = ut * u_scale / s_scale
        #    u_dist, s_dist = self.get_likelihood(ut, st, u_log_library, s_log_library, u_scale, s_scale, u_read_depth=None, s_read_depth=None)
        #    u = pyro.sample("u", u_dist, obs=u_obs)
        #    s = pyro.sample("s", s_dist, obs=s_obs)
        return ut, st


class MultiKineticsModelDirichlet(VelocityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.k)
        if self.guide_type in [
            "velocity_auto",
            "velocity_auto_depth",
            "velocity_auto_t0_constraint",
        ]:
            self.time_encoder = TimeEncoder2(
                self.num_genes,
                n_output=1,
                dropout_rate=0.5,
                activation_fn=nn.ELU,
                n_layers=3,
                var_eps=1e-6,
            )

    def get_time(
        self,
        u_scale,
        s_scale,
        alpha,
        beta,
        gamma,
        u_obs,
        s_obs,
        u0,
        s0,
        t0,
        dt_switching,
        u_inf,
        s_inf,
        u_read_depth=None,
        s_read_depth=None,
    ):
        u_ = u_obs
        s_ = s_obs
        s_scale = std_u = self.one
        tau = tau_inv(u_, s_, u0, s0, alpha, beta, gamma)
        ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
        if self.cell_specific_kinetics is None:
            tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
            ut_, st_ = mRNA(tau_, u_inf, s_inf, self.zero, beta, gamma)
        state_on = ((ut - u_) / std_u) ** 2 + ((st - s_) / s_scale) ** 2
        state_zero = ((ut - u0) / std_u) ** 2 + ((st - s0) / s_scale) ** 2
        if self.cell_specific_kinetics is None:
            state_inf = ((ut_ - u_inf) / std_u) ** 2 + (
                (st_ - s_inf) / s_scale
            ) ** 2
            state_off = ((ut_ - u_) / std_u) ** 2 + ((st_ - s_) / s_scale) ** 2
            cell_gene_state_logits = torch.stack(
                [state_on, state_zero, state_off, state_inf], dim=-1
            ).argmin(-1)
        if self.cell_specific_kinetics is None:
            state = (cell_gene_state_logits > 1) == self.zero
            t = torch.where(state, tau + t0, tau_ + dt_switching + t0)
        else:
            t = softplus(tau + t0)
        cell_time_loc, cell_time_scale = self.time_encoder(t)
        t = pyro.sample(
            "cell_time", LogNormal(cell_time_loc, torch.sqrt(cell_time_scale))
        )
        return t

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        u_log_library_mean = tensor_dict["u_lib_size_mean"]
        s_log_library_mean = tensor_dict["s_lib_size_mean"]
        u_log_library_scale = tensor_dict["u_lib_size_scale"]
        s_log_library_scale = tensor_dict["s_lib_size_scale"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        if "pyro_cell_state" in tensor_dict:
            cell_state = tensor_dict["pyro_cell_state"]
        else:
            cell_state = None
        if "time_info" in tensor_dict:
            time_info = tensor_dict["time_info"]
        else:
            time_info = None
        return (
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_mean,
            s_log_library_mean,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        ), {}

    def create_plates(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        kinetics_plate = pyro.plate("kinetics", self.k, dim=-2)
        return cell_plate, gene_plate, kinetics_plate

    def get_rna(
        self,
        alpha,
        beta,
        gamma,
        t,
        u0,
        s0,
        t0,
        switching=None,
        u_inf=None,
        s_inf=None,
    ):
        state = (
            pyro.sample(
                "cell_gene_state",
                Bernoulli(logits=t - switching),
                infer={"enumerate": "parallel"},
            )
            == self.zero
        )
        ##infer={'enumerate': 'sequential'}) == self.zero
        alpha_off = self.zero
        u0_vec = torch.where(state, u0, u_inf)
        s0_vec = torch.where(state, s0, s_inf)
        alpha_vec = torch.where(state, alpha, alpha_off)
        tau = softplus(torch.where(state, t - t0, t - switching))
        return mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_plate, gene_plate, kinetics_plate = self.create_plates(
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_loc,
            s_log_library_loc,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        )

        ##with gene_plate:
        ##    if (self.likelihood == 'Normal') or (self.guide_type == 'auto'):
        ##        u_scale = self.u_scale
        ##        s_scale = self.one
        ##        if self.likelihood == 'Normal':
        ##            s_scale = self.s_scale
        ##    else:
        ##        # NegativeBinomial and Poisson model
        u_scale = s_scale = self.one

        t0 = pyro.sample("t0", Normal(self.one, self.one))

        with kinetics_plate, gene_plate:
            alpha_k = pyro.sample("alpha", LogNormal(self.zero, self.one))
            beta_k = pyro.sample("beta", LogNormal(self.zero, self.one))
            gamma_k = pyro.sample("gamma", LogNormal(self.zero, self.one))
            dt_switching_k = pyro.sample(
                "dt_switching", LogNormal(self.zero, self.one)
            )
            u_inf_k, s_inf_k = mRNA(
                dt_switching_k, self.zero, self.zero, alpha_k, beta_k, gamma_k
            )
            ###u_inf_k, s_inf_k = mRNA(dt_switching, self.zero, self.zero, alpha_k, beta_k, gamma_k)
            ##switching_k = t0_k + dt_switching_k
            switching_k = t0 + dt_switching_k
            # assert switching_k.shape == (kinetics_plate.size, gene_plate.size)
            # assert u_inf_k.shape == (kinetics_plate.size, gene_plate.size)
            if self.add_offset:
                u0_k = pyro.sample("u_offset", LogNormal(self.zero, self.one))
                s0_k = pyro.sample("s_offset", LogNormal(self.zero, self.one))
            else:
                s0 = u0 = self.zero

        # Cell kinetics probability
        with cell_plate:
            kinetics_probs = pyro.sample(
                "kinetics_prob", Dirichlet(self.one.new_ones(self.k) / self.k)
            )
            assert kinetics_probs.shape == (
                cell_plate.subsample_size,
                1,
                kinetics_plate.size,
            ), kinetics_probs.shape
            kinetics_lineage = pyro.sample(
                "kinetics_lineage",
                Categorical(kinetics_probs),
                infer={"enumerate": "parallel"},
            )
            ##infer={'enumerate': 'sequential'})
            # assert kinetics_lineage.shape == (cell_plate.subsample_size, 1), kinetics_lineage.shape

            t = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )
            if self.correct_library_size and (self.likelihood != "Normal"):
                u_read_depth = pyro.sample(
                    "u_read_depth",
                    LogNormal(u_log_library, u_log_library_scale),
                )
                s_read_depth = pyro.sample(
                    "s_read_depth",
                    LogNormal(s_log_library, s_log_library_scale),
                )
                ##if self.correct_library_size == 'cell_size_regress':
                ##    # cell-wise coef per cell
                ##    u_cell_size_coef = pyro.sample("u_cell_size_coef", Normal(self.zero, self.one))
                ##    ut_coef = pyro.sample("ut_coef", Normal(self.zero, self.one))
                ##    s_cell_size_coef = pyro.sample("s_cell_size_coef", Normal(self.zero, self.one))
                ##    st_coef = pyro.sample("st_coef", Normal(self.zero, self.one))
                ##else:
                ##    u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
            else:
                u_read_depth = s_read_depth = None
                u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None

            alpha = Vindex(
                alpha_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            beta = Vindex(beta_k.transpose(kinetics_plate.dim, gene_plate.dim))[
                ..., kinetics_lineage
            ]
            gamma = Vindex(
                gamma_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            switching = Vindex(
                switching_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            ##t0 = Vindex(t0_k.transpose(kinetics_plate.dim, gene_plate.dim))[..., kinetics_lineage]
            u_inf = Vindex(
                u_inf_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            s_inf = Vindex(
                s_inf_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]

            if self.add_offset:
                u0 = Vindex(u0_k.transpose(kinetics_plate.dim, gene_plate.dim))[
                    ..., kinetics_lineage
                ]
                s0 = Vindex(s0_k.transpose(kinetics_plate.dim, gene_plate.dim))[
                    ..., kinetics_lineage
                ]
            # print(s_inf.shape)
            # assert s_inf.shape == (cell_plate.subsample_size, gene_plate.size), s_inf.shape

        with cell_plate, gene_plate:
            if (
                self.guide_type == "auto_t0_constraint"
                or self.guide_type == "velocity_auto_t0_constraint"
            ):
                pyro.sample(
                    "time_constraint", Bernoulli(logits=t - t0), obs=self.one
                )
            ut, st = self.get_rna(
                alpha, beta, gamma, t, u0, s0, t0, switching, u_inf, s_inf
            )
            u_dist, s_dist = self.get_likelihood(
                ut * u_scale / s_scale,
                st,
                u_log_library,
                s_log_library,
                None,
                None,
                u_read_depth=u_read_depth,
                s_read_depth=s_read_depth,
            )
            u = pyro.sample("u", u_dist, obs=u_obs)
            s = pyro.sample("s", s_dist, obs=s_obs)
            pyro.deterministic("alpha_k", alpha, event_dim=0)
            pyro.deterministic("beta_k", beta, event_dim=0)
            pyro.deterministic("gamma_k", gamma, event_dim=0)
            pyro.deterministic("switching_k", switching, event_dim=0)
            pyro.deterministic("t0_k", t0, event_dim=0)


class MultiKineticsMultiSwitchModel(VelocityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.k)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        u_log_library_mean = tensor_dict["u_lib_size_mean"]
        s_log_library_mean = tensor_dict["s_lib_size_mean"]
        u_log_library_scale = tensor_dict["u_lib_size_scale"]
        s_log_library_scale = tensor_dict["s_lib_size_scale"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        if "pyro_cell_state" in tensor_dict:
            cell_state = tensor_dict["pyro_cell_state"]
        else:
            cell_state = None
        if "time_info" in tensor_dict:
            time_info = tensor_dict["time_info"]
        else:
            time_info = None
        return (
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_mean,
            s_log_library_mean,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        ), {}

    def create_plates(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        kinetics_plate = pyro.plate("kinetics", self.k, dim=-2)
        return cell_plate, gene_plate, kinetics_plate

    def get_rna(
        self,
        alpha,
        beta,
        gamma,
        t,
        u0,
        s0,
        t0,
        switching=None,
        u_inf=None,
        s_inf=None,
    ):
        state = (
            pyro.sample(
                "cell_gene_state",
                Bernoulli(logits=t - switching),
                # infer={'enumerate': 'parallel'}) == self.zero
                infer={"enumerate": "sequential"},
            )
            == self.zero
        )
        alpha_off = self.zero
        u0_vec = torch.where(state, u0, u_inf)
        s0_vec = torch.where(state, s0, s_inf)
        alpha_vec = torch.where(state, alpha, alpha_off)
        tau = softplus(torch.where(state, t - t0, t - switching))
        return mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_plate, gene_plate, kinetics_plate = self.create_plates(
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_loc,
            s_log_library_loc,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        )

        t0 = pyro.sample("t0", LogNormal(self.zero, self.one))
        with kinetics_plate, gene_plate:
            alpha_k = pyro.sample("alpha", LogNormal(self.zero, self.one))
            beta_k = pyro.sample("beta", LogNormal(self.zero, self.one))
            gamma_k = pyro.sample("gamma", LogNormal(self.zero, self.one))
            dt_switching_k = pyro.sample(
                "dt_switching", LogNormal(self.zero, self.one)
            )
            u_inf_k, s_inf_k = mRNA(
                dt_switching_k, self.zero, self.zero, alpha_k, beta_k, gamma_k
            )
            switching_k = t0 + dt_switching_k
            assert switching_k.shape == (kinetics_plate.size, gene_plate.size)

        # Cell kinetics probability
        with cell_plate:
            t = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )
            u_read_depth = pyro.sample(
                "u_read_depth", LogNormal(u_log_library, u_log_library_scale)
            )
            s_read_depth = pyro.sample(
                "s_read_depth", LogNormal(s_log_library, s_log_library_scale)
            )
            if (
                self.guide_type == "auto_t0_constraint"
                or self.guide_type == "velocity_auto_t0_constraint"
            ):
                pyro.sample(
                    "time_constraint", Bernoulli(logits=t - t0), obs=self.one
                )  # start point constraint
                with gene_plate:  # end point constraint
                    pyro.sample(
                        "time_constraint2",
                        Bernoulli(logits=switching_k[-1] - t),
                        obs=self.one,
                    )

        with cell_plate, gene_plate:
            kinetics_logits = switching_k.transpose(
                kinetics_plate.dim, gene_plate.dim
            ).unsqueeze(-2)
            assert kinetics_logits.shape == (
                1,
                gene_plate.size,
                kinetics_plate.size,
            )
            kinetics_logits = t.unsqueeze(-1) - kinetics_logits
            assert kinetics_logits.shape == (
                cell_plate.subsample_size,
                gene_plate.size,
                kinetics_plate.size,
            )

            kinetics_lineage = pyro.sample(
                "kinetics_lineage",
                Categorical(kinetics_logits),
                # infer={'enumerate': 'parallel'})
                infer={"enumerate": "sequential"},
            )

            assert kinetics_lineage.shape == (
                cell_plate.subsample_size,
                gene_plate.size,
            ), kinetics_lineage.shape
            alpha = Vindex(
                alpha_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            beta = Vindex(beta_k.transpose(kinetics_plate.dim, gene_plate.dim))[
                ..., kinetics_lineage
            ]
            gamma = Vindex(
                gamma_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            switching = Vindex(
                switching_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            u_inf = Vindex(
                u_inf_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]
            s_inf = Vindex(
                s_inf_k.transpose(kinetics_plate.dim, gene_plate.dim)
            )[..., kinetics_lineage]

            tau = softplus(kinetics_logits)
            ##ut, st = mRNA(tau, self.zero, self.zero, alpha, beta, gamma)

            u_dist, s_dist = self.get_likelihood(
                ut,
                st,
                u_log_library,
                s_log_library,
                None,
                None,
                u_read_depth=u_read_depth,
                s_read_depth=s_read_depth,
            )
            u = pyro.sample("u", u_dist, obs=u_obs)
            s = pyro.sample("s", s_dist, obs=s_obs)
            pyro.deterministic("alpha_k", alpha, event_dim=0)
            pyro.deterministic("beta_k", beta, event_dim=0)
            pyro.deterministic("gamma_k", gamma, event_dim=0)
            pyro.deterministic("switching_k", switching, event_dim=0)
            pyro.deterministic("t0_k", t0, event_dim=0)


class MultiKineticsModelDirichletLinear(VelocityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.k)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        u_log_library_mean = tensor_dict["u_lib_size_mean"]
        s_log_library_mean = tensor_dict["s_lib_size_mean"]
        u_log_library_scale = tensor_dict["u_lib_size_scale"]
        s_log_library_scale = tensor_dict["s_lib_size_scale"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        if "pyro_cell_state" in tensor_dict:
            cell_state = tensor_dict["pyro_cell_state"]
        else:
            cell_state = None
        if "time_info" in tensor_dict:
            time_info = tensor_dict["time_info"]
        else:
            time_info = None
        return (
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_mean,
            s_log_library_mean,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        ), {}

    def get_time(
        self,
        u_scale,
        s_scale,
        alpha,
        beta,
        gamma,
        u_obs,
        s_obs,
        u0,
        s0,
        t0,
        dt_switching,
        u_inf,
        s_inf,
        u_read_depth=None,
        s_read_depth=None,
    ):
        u_ = u_obs
        s_ = s_obs
        std_u = u_scale / scale
        tau = tau_inv(u_, s_, u0, s0, alpha, beta, gamma)
        ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
        if self.cell_specific_kinetics is None:
            tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
            ut_, st_ = mRNA(tau_, u_inf, s_inf, self.zero, beta, gamma)
        state_on = ((ut - u_) / std_u) ** 2 + ((st - s_) / s_scale) ** 2
        state_zero = ((ut - u0) / std_u) ** 2 + ((st - s0) / s_scale) ** 2
        if self.cell_specific_kinetics is None:
            state_inf = ((ut_ - u_inf) / std_u) ** 2 + (
                (st_ - s_inf) / s_scale
            ) ** 2
            state_off = ((ut_ - u_) / std_u) ** 2 + ((st_ - s_) / s_scale) ** 2
            cell_gene_state_logits = torch.stack(
                [state_on, state_zero, state_off, state_inf], dim=-1
            ).argmin(-1)
        if self.cell_specific_kinetics is None:
            state = (cell_gene_state_logits > 1) == self.zero
            t = torch.where(state, tau + t0, tau_ + dt_switching + t0)
        else:
            t = softplus(tau + t0)
        cell_time_loc, cell_time_scale = self.time_encoder(t)
        t = pyro.sample(
            "cell_time", LogNormal(cell_time_loc, torch.sqrt(cell_time_scale))
        )
        return t

    def create_plates(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        kinetics_plate = pyro.plate("kinetics", self.k, dim=-3)
        return cell_plate, gene_plate, kinetics_plate

    def get_rna(
        self,
        alpha,
        beta,
        gamma,
        t,
        u0,
        s0,
        t0,
        switching=None,
        u_inf=None,
        s_inf=None,
    ):
        state = (
            pyro.sample(
                "cell_gene_state",
                Bernoulli(logits=t - switching),
                infer={"enumerate": "parallel"},
            )
            == self.zero
        )
        alpha_off = self.zero
        u0_vec = torch.where(state, u0, u_inf)
        s0_vec = torch.where(state, s0, s_inf)
        alpha_vec = torch.where(state, alpha, alpha_off)
        tau = softplus(torch.where(state, t - t0, t - switching))
        return mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # This piece of codes work for both sequential and parallel enumeration
        cell_plate, gene_plate, kinetics_plate = self.create_plates(
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_loc,
            s_log_library_loc,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        )
        with kinetics_plate, gene_plate:
            alpha_k = pyro.sample("alpha", LogNormal(self.zero, self.one))
            beta_k = pyro.sample("beta", LogNormal(self.zero, self.one))
            gamma_k = pyro.sample("gamma", LogNormal(self.zero, self.one))
            dt_switching_k = pyro.sample(
                "dt_switching", LogNormal(self.zero, self.one)
            )
            t0_k = pyro.sample("t0", Normal(self.one, self.one))
            u_inf_k, s_inf_k = mRNA(
                dt_switching_k, self.zero, self.zero, alpha_k, beta_k, gamma_k
            )
            switching_k = t0_k + dt_switching_k
            assert switching_k.shape == (
                kinetics_plate.size,
                1,
                gene_plate.size,
            )
            # assert u_inf_k.shape == (kinetics_plate.size, gene_plate.size)
            if self.add_offset:
                u_offset_k = pyro.sample(
                    "u_offset", LogNormal(self.zero, self.one)
                )
                s_offset_k = pyro.sample(
                    "s_offset", LogNormal(self.zero, self.one)
                )
            else:
                u_offset_k = s_offset_k = self.zero

        with kinetics_plate:  # kinetics specific weights
            kinetics_weights = pyro.sample(
                "kinetics_weights", Beta(self.one * 2, self.one * 5)
            )
            assert kinetics_weights.shape == (
                kinetics_plate.size,
                1,
                1,
            ), kinetics_weights.shape

        ##    assert kinetics_weights.shape == (kinetics_plate.size, 1, gene_plate.size), kinetics_weights.shape

        # Cell kinetics probability
        with cell_plate:
            t = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )
            if self.correct_library_size and (self.likelihood != "Normal"):
                u_read_depth = pyro.sample(
                    "u_read_depth",
                    LogNormal(u_log_library, u_log_library_scale),
                )
                s_read_depth = pyro.sample(
                    "s_read_depth",
                    LogNormal(s_log_library, s_log_library_scale),
                )
                ##if self.correct_library_size == 'cell_size_regress':
                ##    # cell-wise coef per cell
                ##    u_cell_size_coef = pyro.sample("u_cell_size_coef", Normal(self.zero, self.one))
                ##    ut_coef = pyro.sample("ut_coef", Normal(self.zero, self.one))
                ##    s_cell_size_coef = pyro.sample("s_cell_size_coef", Normal(self.zero, self.one))
                ##    st_coef = pyro.sample("st_coef", Normal(self.zero, self.one))
                ##else:
                ##    u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
            else:
                u_read_depth = s_read_depth = None
                u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None

            with gene_plate:
                with kinetics_plate:
                    if (
                        self.guide_type == "auto_t0_constraint"
                        or self.guide_type == "velocity_auto_t0_constraint"
                    ):
                        pyro.sample(
                            "time_constraint",
                            Bernoulli(logits=t - t0_k),
                            obs=self.one,
                        )

                    logits_k = t - switching_k
                    assert logits_k.shape == (
                        kinetics_plate.size,
                        cell_plate.subsample_size,
                        gene_plate.size,
                    )
                    state_k = (
                        pyro.sample(
                            "cell_gene_state",
                            Bernoulli(logits=logits_k),
                            infer={"enumerate": "sequential"},
                        )
                        == self.zero
                    )
                    # infer={'enumerate': 'parallel'}) == self.zero
                    # assert state_k.shape == (kinetics_plate.size, cell_plate.subsample_size, gene_plate.size)
                    u0_k = torch.where(state_k, u_offset_k, u_inf_k)
                    s0_k = torch.where(state_k, s_offset_k, s_inf_k)

                    alpha_k = torch.where(state_k, alpha_k, self.zero)
                    tau_k = softplus(torch.where(state_k, t - t0_k, logits_k))
                    ut_k, st_k = mRNA(
                        tau_k, u0_k, s0_k, alpha_k, beta_k, gamma_k
                    )
                    # assert ut_k.shape == (kinetics_plate.size, cell_plate.subsample_size, gene_plate.size), ut_k.shape

                    # torch.Size([2, 2, 2930, 2000]) parallel not apply for einsum, torch.Size([2, 2930, 2000]) sequential
                    # (cell_plate.subsample_size, gene_plate.size, kinetics_plate.size) @ (kinetics_plate.size, 1, 1)
                    # print(ut_k.size())
                    ###if len(ut_k.size()) == 3:
                    # ut = ut_k.permute(1, 2, 0) @ kinetics_weights.squeeze(-2)
                    # st = st_k.permute(1, 2, 0) @ kinetics_weights.squeeze(-2)
                    # assert ut.shape == (cell_plate.subsample_size, gene_plate.size, 1), ut.shape
                    # below only apply to sequential
                    ut = torch.einsum(
                        "...ijk,ijk->...jk", ut_k, kinetics_weights
                    ).unsqueeze(-3)
                    st = torch.einsum(
                        "...ijk,ijk->...jk", st_k, kinetics_weights
                    ).unsqueeze(-3)
                    # assert ut.shape == (cell_plate.subsample_size, gene_plate.size), ut.shape
                    ###else:
                    ###    #assert ut_k.shape == (kinetics_plate.size, kinetics_plate.size, cell_plate.subsample_size, gene_plate.size, 1), ut.shape
                    ###    ut = ut_k.permute(0, 2, 3, 1) @ kinetics_weights.squeeze(-2)
                    ###    st = st_k.permute(0, 2, 3, 1) @ kinetics_weights.squeeze(-2)
                    ###ut = ut.squeeze()
                    ###st = st.squeeze()
                # print(ut.shape)
                u_dist, s_dist = self.get_likelihood(
                    ut,
                    st,
                    u_log_library,
                    s_log_library,
                    None,
                    None,
                    u_read_depth=u_read_depth,
                    s_read_depth=s_read_depth,
                )
                u = pyro.sample("u", u_dist, obs=u_obs)
                s = pyro.sample("s", s_dist, obs=s_obs)
        with gene_plate:
            beta = torch.einsum("ijk,ijk->jk", beta_k, kinetics_weights)
            gamma = torch.einsum("ijk,ijk->jk", gamma_k, kinetics_weights)
            pyro.deterministic("beta_k", beta, event_dim=0)
            pyro.deterministic("gamma_k", gamma, event_dim=0)

        ### remove kinetics plate using to_event
        ### put kinetics_weights into event_shape
        ### only work for sequential enumeration
        # cell_plate, gene_plate = self.create_plates(u_obs, s_obs, u_log_library, s_log_library, u_log_library_loc, s_log_library_loc, u_log_library_scale, s_log_library_scale, ind_x, cell_state, time_info)
        # zero = self.zero.new_zeros(self.k)
        # one = self.one.new_ones(self.k)
        # with gene_plate:
        #    alpha_k = pyro.sample("alpha", LogNormal(zero, one).to_event(1))
        #    beta_k = pyro.sample("beta", LogNormal(zero, one).to_event(1))
        #    gamma_k = pyro.sample("gamma", LogNormal(zero, one).to_event(1))
        #    dt_switching_k = pyro.sample("dt_switching", LogNormal(zero, one).to_event(1))
        #    t0_k = pyro.sample("t0", Normal(one, one).to_event(1))
        #    u_inf_k, s_inf_k = mRNA(dt_switching_k, zero, zero, alpha_k, beta_k, gamma_k)
        #    switching_k = t0_k + dt_switching_k
        #    assert u_inf_k.shape == (gene_plate.size, self.k), u_inf_k.shape
        #    if self.add_offset:
        #        u_offset_k = pyro.sample("u_offset", LogNormal(zero, one).to_event(1))
        #        s_offset_k = pyro.sample("s_offset", LogNormal(zero, one).to_event(1))
        #    else:
        #        u_offset_k = s_offset_k = zero

        # kinetics_weights = pyro.sample("kinetics_weights", Beta(one*2, one*5).to_event(1))
        # assert kinetics_weights.shape == (self.k, ), kinetics_weights.shape
        # with cell_plate:
        #    t = pyro.sample("cell_time", LogNormal(self.zero, self.one).mask(self.include_prior))
        #    if self.correct_library_size and (self.likelihood != 'Normal'):
        #        u_read_depth = pyro.sample("u_read_depth", LogNormal(u_log_library, u_log_library_scale))
        #        s_read_depth = pyro.sample("s_read_depth", LogNormal(s_log_library, s_log_library_scale))
        #        ##if self.correct_library_size == 'cell_size_regress':
        #        ##    # cell-wise coef per cell
        #        ##    u_cell_size_coef = pyro.sample("u_cell_size_coef", Normal(self.zero, self.one))
        #        ##    ut_coef = pyro.sample("ut_coef", Normal(self.zero, self.one))
        #        ##    s_cell_size_coef = pyro.sample("s_cell_size_coef", Normal(self.zero, self.one))
        #        ##    st_coef = pyro.sample("st_coef", Normal(self.zero, self.one))
        #        ##else:
        #        ##    u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
        #    else:
        #        u_read_depth = s_read_depth = None
        #        u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
        #    with gene_plate:
        #        ###if self.guide_type == 'auto_t0_constraint' or self.guide_type == 'velocity_auto_t0_constraint':
        #        ###    pyro.sample("time_constraint", Bernoulli(logits=t-t0_k), obs=self.one)
        #        logits_k = t.unsqueeze(-1)-switching_k
        #        assert logits_k.shape == (cell_plate.subsample_size, gene_plate.size, self.k)

        #        state_k = pyro.sample("cell_gene_state", Bernoulli(logits=logits_k).to_event(1),
        #                              infer={'enumerate': 'sequential'}) == self.zero
        #                              #infer={'enumerate': 'parallel'}) == self.zero # NotImplementedError: Enumeration over cartesian product is not implemented
        #        assert state_k.shape == (cell_plate.subsample_size, gene_plate.size, self.k)

        #        u0_k = torch.where(state_k, u_offset_k, u_inf_k)
        #        s0_k = torch.where(state_k, s_offset_k, s_inf_k)

        #        alpha_k = torch.where(state_k, alpha_k, self.zero)
        #        tau_k = softplus(torch.where(state_k, t.unsqueeze(-1)-t0_k, logits_k))
        #        ut_k, st_k = mRNA(tau_k, u0_k, s0_k, alpha_k, beta_k, gamma_k)
        #        ut = torch.einsum("ijk,k->ij", ut_k, kinetics_weights)
        #        st = torch.einsum("ijk,k->ij", st_k, kinetics_weights)
        #        assert ut.shape == (cell_plate.subsample_size, gene_plate.size), ut.shape
        #        u_dist, s_dist = self.get_likelihood(ut, st, u_log_library, s_log_library, None, None, u_read_depth=u_read_depth, s_read_depth=s_read_depth)
        #        u = pyro.sample("u", u_dist, obs=u_obs)
        #        s = pyro.sample("s", s_dist, obs=s_obs)


class AuxTrajectoryModel(PyroModule):
    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        likelihood: str = "Normal",
        num_aux_cells=30,
        **initial_values,
    ):
        assert num_cells > 0 and num_genes > 0
        super().__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.n_obs = None

        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("one", torch.tensor(1.0))
        self.likelihood = likelihood
        self.num_aux_cells = num_aux_cells
        for key in initial_values:
            self.register_buffer(f"{key}_init", initial_values[key])

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        # u_obs = tensor_dict[VelocityCONSTANTS.U_KEY]
        # s_obs = tensor_dict[VelocityCONSTANTS.X_KEY]
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        return (u_obs, s_obs, u_log_library, s_log_library, ind_x), {}

    def create_plates(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        return cell_plate, gene_plate

    def forward(self, u_obs, s_obs, u_log_library, s_log_library, ind_x):
        cell_plate, gene_plate = self.create_plates(
            u_obs, s_obs, u_log_library, s_log_library, ind_x
        )
        with gene_plate, poutine.mask(mask=True):
            alpha = pyro.sample("alpha", LogNormal(self.zero, self.one))
            beta = pyro.sample("beta", LogNormal(self.zero, self.one))
            gamma = pyro.sample("gamma", LogNormal(self.zero, self.one))
            u_scale = pyro.sample("u_scale", LogNormal(self.zero, self.one))
            s_scale = pyro.sample("s_scale", LogNormal(self.zero, self.one))
            switching = pyro.sample("switching", LogNormal(self.zero, self.one))

        aux_cell_plate = pyro.plate("aux_cells", self.num_aux_cells, dim=-2)
        u_prev = s_prev = self.zero
        # u_aux = pyro.param("u_aux", lambda: torch.zeros(aux_cell_plate.size, gene_plate.size), constraint=positive).to(u_obs.device)
        # s_aux = pyro.param("s_aux", lambda: torch.zeros(aux_cell_plate.size, gene_plate.size), constraint=positive).to(u_obs.device)
        # dt_aux = pyro.param("dt_aux", lambda: torch.zeros(aux_cell_plate.size), constraint=positive).to(u_obs.device)
        # s_aux = torch.zeros(aux_cell_plate.size, gene_plate.size).to(u_obs.device)
        # u_aux = torch.zeros(aux_cell_plate.size, gene_plate.size).to(u_obs.device)
        # dt_aux = torch.zeros(aux_cell_plate.size, gene_plate.size).to(u_obs.device)
        rate = 1.0 / aux_cell_plate.size  # TODO make time dependent
        u_aux = []
        s_aux = []
        dt_aux = []
        for step in pyro.markov(range(aux_cell_plate.size)):
            # dt_aux[step] += pyro.sample(f"dt_aux_{step}", dist.Exponential(rate))
            dt_aux.append(pyro.sample(f"dt_aux_{step}", dist.Exponential(rate)))
            with gene_plate:
                u_aux_noise = pyro.sample(
                    f"u_aux_noise_{step}", Normal(self.zero, self.one)
                )
                s_aux_noise = pyro.sample(
                    f"s_aux_noise_{step}", Normal(self.zero, self.one)
                )
                state_aux = (
                    pyro.sample(
                        f"cell_gene_state_{step}",
                        Bernoulli(logits=dt_aux[step] - switching),
                    )
                    == self.zero
                )
                alpha = torch.where(state_aux, alpha, self.zero)
                u_loc, s_loc = mRNA(
                    dt_aux[step], u_prev, s_prev, alpha, beta, gamma
                )
                # u_aux[step] = u_loc + u_noise * torch.sqrt(2 * dt_aux[step])
                # s_aux[step] = s_loc + s_noise * torch.sqrt(2 * dt_aux[step])
                u_prev = u_loc + u_aux_noise * torch.sqrt(2 * dt_aux[step])
                s_prev = s_loc + s_aux_noise * torch.sqrt(2 * dt_aux[step])
                u_aux.append(u_prev)
                s_aux.append(s_prev)

        u_aux = torch.vstack(u_aux).to(u_obs.device)
        s_aux = torch.vstack(s_aux).to(u_obs.device)
        dt_aux = torch.hstack(dt_aux).to(u_obs.device)
        # print(u_aux.shape, dt_aux.shape)

        with aux_cell_plate, gene_plate:
            u_aux_obs = pyro.sample(
                "u_aux_obs", Normal(u_aux, u_scale), obs=self.aux_u_obs_init
            )
            s_aux_obs = pyro.sample(
                "s_aux_obs", Normal(s_aux, s_scale), obs=self.aux_s_obs_init
            )

        # approximation: real cells independent
        prob_real_to_aux = (
            torch.ones(self.num_aux_cells) / self.num_aux_cells
        ).to(u_obs.device)
        with cell_plate:
            order = pyro.sample(
                "order",
                Categorical(prob_real_to_aux),
                infer={"enumerate": "parallel"},
            )
            dt = dt_aux[..., order]
            with gene_plate:
                u_noise = pyro.sample("u_noise", Normal(self.zero, self.one))
                s_noise = pyro.sample("s_noise", Normal(self.zero, self.one))
                ###state = pyro.sample("cell_gene_state", Bernoulli(probs=torch.sigmoid(t[c] - switching_start) * (1 - torch.sigmoid(switching_stop - t[c]))
                state = (
                    pyro.sample(
                        "cell_gene_state", Bernoulli(logits=dt - switching)
                    )
                    == self.zero
                )
                alpha = torch.where(state, alpha, self.zero)
                u_prev = u_aux[..., order, :].squeeze()
                s_prev = s_aux[..., order, :].squeeze()
                u_loc, s_loc = mRNA(dt, u_prev, s_prev, alpha, beta, gamma)
                u = u_loc + u_noise * torch.sqrt(2 * dt)
                s = s_loc + s_noise * torch.sqrt(2 * dt)
                u_obs = pyro.sample(
                    "u_obs", Normal(u * u_scale / s_scale, u_scale), obs=u_obs
                )
                s_obs = pyro.sample("s_obs", Normal(s, s_scale), obs=s_obs)


# class CellGenerator(PyroModule):
#    @PyroSample
#    def latent_time(self):
#        if self.shared_time:
#            if self.plate_size == 2:
#                return Normal(self.zero, self.one*0.1)  #.mask(False) # with shared cell_time
#            else:
#                return Normal(self.zero, self.one*0.1).expand((self.num_genes, )).to_event(1)  #.mask(False) # with shared cell_time
#        if self.plate_size == 2:
#            return LogNormal(self.zero, self.one).mask(self.include_prior) # without shared cell_time
#        return LogNormal(self.zero, self.one).expand((self.num_genes, )).to_event(1) #.mask(False) # without shared cell_time
#
#    @PyroSample
#    def cell_time(self):
#        if self.shared_time:
#            return Normal(self.zero, self.one).mask(False) # mask=False generate the same estimation as initialization
#        return LogNormal(self.zero, self.one) #.mask(False) # mask=False with LogNormal makes negative correlation
#
#    def forward(
#        self,
#        gene_plate,
#        alpha,
#        beta,
#        gamma,
#        switching,
#        u_inf,
#        s_inf,
#        u_scale,
#        s_scale,
#        u_log_library,
#        s_log_library,
#        u_pcs_mean=None,
#        s_pcs_mean=None,
#        velocity_genecellpair=None
#    ):
#        if self.latent_factor == 'linear':
#            cell_codebook = self.cell_codebook
#            with poutine.mask(mask=self.include_prior):
#                cell_code = self.cell_code
#        if self.shared_time:
#            cell_time = self.cell_time
#
#        with gene_plate:
#            t = self.latent_time
#            #t = pyro.sample("latent_time", LogNormal(self.zero, self.one).mask(self.include_prior))
#            if self.shared_time:
#                if self.t_scale_on:
#                    t = cell_time * t_scale + t
#                else:
#                    t = cell_time + t
#            state = pyro.sample("cell_gene_state", Bernoulli(logits=t-switching)) == self.zero
#            u0_vec = torch.where(state, self.zero, u_inf)
#            s0_vec = torch.where(state, self.zero, s_inf)
#            alpha_vec = torch.where(state, alpha, self.zero)
#            tau = torch.where(state, t, t - switching).clamp(0.)
#            ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
#            #print(tau.shape, ut.shape, st.shape)
#            if self.latent_factor == 'linear':
#                regressor_output = torch.einsum("abc,cd->ad", cell_code, cell_codebook.squeeze())
#                regressor_u = softplus(regressor_output[..., :self.num_genes].squeeze() + u_pcs_mean)
#                regressor_s = softplus(regressor_output[..., self.num_genes:].squeeze() + s_pcs_mean)
#            if self.latent_factor_operation == 'selection':
#                ut = torch.where(velocity_genecellpair == self.one, ut * u_scale / s_scale, softplus(regressor_u))
#                st = torch.where(velocity_genecellpair == self.one, st, softplus(regressor_s))
#            elif self.latent_factor_operation == 'sum':
#                ut = (ut * u_scale / s_scale + regressor_u).clamp(0.)
#                st = (st + regressor_s).clamp(0.)
#            else:
#                ut = (ut * u_scale / s_scale).clamp(0.)
#                st = st.clamp(0.)
#            u_dist, s_dist = self.get_likelihood(ut, st, u_log_library, s_log_library, u_scale, s_scale)
#            u = pyro.sample("u", u_dist)
#            s = pyro.sample("s", s_dist)
#        return ut, st


class DecoderTimeModel(LogNormalModel):
    def __init__(
        self, num_cells: int, num_genes: int, likelihood: str = "Poisson"
    ):
        assert num_cells > 0 and num_genes > 0
        super().__init__(num_cells, num_genes, likelihood)
        self.latent_dim = 10

        self.decoder = DecoderSCVI(
            self.latent_dim,
            num_genes,
            n_layers=3,
            n_hidden=128,
        )

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_plate, gene_plate = self.create_plates(
            u_obs, s_obs, u_log_library, s_log_library, ind_x
        )

        with gene_plate:
            switching = self.switching
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            u_inf, s_inf = mRNA(
                switching, self.zero, self.zero, alpha, beta, gamma
            )

            if self.likelihood == "Normal":
                u_scale = self.u_scale
                s_scale = self.s_scale

        with cell_plate:
            z_loc = u_obs.new_zeros(
                torch.Size((u_obs.shape[0], 1, self.latent_dim))
            )
            z_scale = u_obs.new_ones(
                torch.Size((u_obs.shape[0], 1, self.latent_dim))
            )
            z = pyro.sample(
                "latent_time_latent_space",
                dist.Normal(z_loc, z_scale).to_event(1),
            )  # (cells, 1, components)

        z = z.squeeze()
        t_scale, _, t_rate, t_dropout = self.decoder("gene", z, s_log_library)

        with cell_plate, gene_plate:
            state = (
                pyro.sample(
                    "cell_gene_state", Bernoulli(logits=t_scale - switching)
                )
                == self.zero
            )
            u0_vec = torch.where(state, self.zero, u_inf)
            s0_vec = torch.where(state, self.zero, s_inf)
            alpha_vec = torch.where(state, alpha, self.zero)

            tau = softplus(torch.where(state, t_scale, t_scale - switching))
            ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
            u_dist, s_dist = self.get_likelihood(
                ut, st, u_log_library, s_log_library
            )
            u = pyro.sample("u", u_dist, obs=u_obs)
            s = pyro.sample("s", s_dist, obs=s_obs)


class LatentFactor(LogNormalModel):
    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        likelihood: str = "Normal",
        mask: Optional[torch.Tensor] = None,
        plate_size: int = 2,
        latent_factor_size: int = 10,
    ):
        assert num_cells > 0 and num_genes > 0
        super().__init__(num_cells, num_genes, likelihood)
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.mask = mask
        self.plate_size = plate_size
        self.likelihood = likelihood
        self.latent_factor_size = latent_factor_size

    def create_plate(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-1
        )
        return cell_plate

    def create_plates(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        return cell_plate, gene_plate

    def model(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_plate, gene_plate = self.create_plates(
            u_obs, s_obs, u_log_library, s_log_library, ind_x
        )
        cell_codebook_scale = pyro.param(
            "cell_codebook_scale",
            lambda: torch.tensor(0.1),
            constraint=positive,
        ).to(u_obs.device)
        cell_codebook = pyro.sample(
            "cell_codebook",
            Normal(self.zero, cell_codebook_scale)
            .expand((self.latent_factor_size, self.num_genes * 2))
            .to_event(2)
            .mask(False),
        )
        cell_code_scale = pyro.param(
            "cell_code_scale", lambda: torch.tensor(0.1), constraint=positive
        ).to(u_obs.device)
        with cell_plate, poutine.mask(mask=False):
            cell_code = pyro.sample(
                "cell_code",
                Normal(self.zero, cell_code_scale)
                .expand((self.latent_factor_size,))
                .to_event(1),
            )

        with gene_plate, poutine.mask(mask=False):
            u_scale = self.u_scale
            s_scale = self.s_scale
            u_pcs_mean = pyro.sample("u_pcs_mean", Normal(self.zero, self.one))
            s_pcs_mean = pyro.sample("s_pcs_mean", Normal(self.zero, self.one))

        with cell_plate, gene_plate:
            regressor_output = torch.einsum(
                "abc,cd->ad", cell_code, cell_codebook.squeeze()
            )
            regressor_u = regressor_output[..., : self.num_genes] + u_pcs_mean
            regressor_s = regressor_output[..., self.num_genes :] + s_pcs_mean
            u_loc = softplus(regressor_u)
            s_loc = softplus(regressor_s)
            s_obs = pyro.sample("s", Normal(s_loc, s_scale), obs=s_obs)
            u_obs = pyro.sample("u", Normal(u_loc, u_scale), obs=u_obs)
            return u_obs, s_obs

    def model2(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_plate = self.create_plate(
            u_obs, s_obs, u_log_library, s_log_library, ind_x
        )
        cell_codebook_scale = pyro.param(
            "cell_codebook_scale",
            lambda: torch.tensor(0.1),
            constraint=positive,
        ).to(u_obs.device)
        cell_code_scale = pyro.param(
            "cell_code_scale", lambda: torch.tensor(0.1), constraint=positive
        ).to(u_obs.device)
        cell_codebook = pyro.sample(
            "cell_codebook",
            Normal(self.zero, cell_codebook_scale)
            .expand((self.latent_factor_size, self.num_genes * 2))
            .to_event(2),
        )

        u_scale = pyro.sample(
            "u_scale",
            Normal(self.zero, self.one).expand((self.num_genes,)).to_event(1),
        )
        s_scale = pyro.sample(
            "s_scale",
            Normal(self.zero, self.one).expand((self.num_genes,)).to_event(1),
        )

        u_pcs_mean = pyro.sample(
            "u_pcs_mean",
            Normal(self.zero, self.one).expand((self.num_genes,)).to_event(1),
        )
        s_pcs_mean = pyro.sample(
            "s_pcs_mean",
            Normal(self.zero, self.one).expand((self.num_genes,)).to_event(1),
        )

        with cell_plate:
            cell_code = pyro.sample(
                "cell_code",
                Normal(self.zero, cell_code_scale)
                .expand((self.latent_factor_size,))
                .to_event(1),
            )
            regressor_output = torch.einsum(
                "ab,bd->ad", cell_code, cell_codebook.squeeze()
            )
            regressor_u = softplus(
                regressor_output[..., : self.num_genes] + u_pcs_mean
            )
            regressor_s = softplus(
                regressor_output[..., self.num_genes :] + s_pcs_mean
            )
            u_dist, s_dist = self.get_likelihood(
                regressor_u,
                regressor_s,
                u_log_library,
                s_log_library,
                None,
                None,
            )
            u = pyro.sample("u", u_dist.to_event(1), obs=u_obs)
            s = pyro.sample("s", s_dist.to_event(1), obs=s_obs)

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.plate_size == 2:
            self.model(u_obs, s_obs, u_log_library, s_log_library, ind_x)
        else:
            self.model2(u_obs, s_obs, u_log_library, s_log_library, ind_x)


##class VelocityModelAuto(AuxCellVelocityModel):
##    def __init__(self, *args, **kwargs):
##        super().__init__(*args, **kwargs)
##        if self.guide_type in ['velocity_auto', 'velocity_auto_depth', 'velocity_auto_t0_constraint']:
##            self.time_encoder = TimeEncoder2(
##                self.num_genes,
##                n_output=1,
##                dropout_rate=0.5,
##                activation_fn=nn.ELU,
##                n_layers=3, var_eps=1e-6)
##        if self.correct_library_size and (self.guide_type == 'velocity_auto' or self.guide_type == 'velocity_auto_t0_constraint'):
##            self.u_lib_encoder = TimeEncoder2(self.num_genes+1, 1, n_layers=3, dropout_rate=0.5)
##            self.s_lib_encoder = TimeEncoder2(self.num_genes+1, 1, n_layers=3, dropout_rate=0.5)
##        if self.cell_specific_kinetics is not None:
##            self.multikinetics_encoder = TimeEncoder2(
##                    1, # encode cell state
##                    1, # encode cell specificity of kinetics
##                    dropout_rate=0.5,
##                    last_layer_activation=nn.Sigmoid(),
##                    n_layers=3)
##
##    def get_rna(self, u_scale, s_scale,
##                alpha, beta, gamma,
##                t, u0, s0, t0,
##                switching=None, u_inf=None, s_inf=None):
##        if self.cell_specific_kinetics is None:
##            state = pyro.sample("cell_gene_state", Bernoulli(logits=t-switching),
##                                infer={'enumerate': 'sequential'}) == self.zero
##                                ###infer={'enumerate': 'parallel'}) == self.zero
##            u0_vec = torch.where(state, u0, u_inf)
##            s0_vec = torch.where(state, s0, s_inf)
##            alpha_vec = torch.where(state, alpha, self.zero)
##            tau = softplus(torch.where(state, t - t0, t - switching))
##        else:
##            u0_vec = u0
##            s0_vec = s0
##            alpha_vec = alpha
##            tau = softplus(t - t0)
##            ##tau = relu(torch.where(state, t - t0, t - switching))
##        ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
##        ut = ut * u_scale / s_scale
##        return ut, st
##
##    def get_time(self,
##                 u_scale, s_scale,
##                 alpha, beta, gamma,
##                 u_obs, s_obs, u0, s0, t0,
##                 dt_switching, u_inf, s_inf,
##                 u_read_depth=None, s_read_depth=None,
##                 ut_coef=None,
##                 u_cell_size_coef=None,
##                 s_cell_size_coef=None,
##                 st_coef=None):
##        scale = u_scale / s_scale
##
##        #if u_read_depth is None:
##        u_ = u_obs / scale
##        s_ = s_obs
##        #else:
##        #    #neural network correction of read depth not converge
##        #    #if self.model.correct_library_size == 'cell_size_regress'
##        #    #    u_ = (torch.log(u_obs+self.one*1e-6)-u_cell_size_coef*u_read_depth)/scale
##        #    #    s_ = (torch.log(s_obs+self.one*1e-6)-s_cell_size_coef*s_read_depth)
##        #    #else:
##        #    u_ = u_obs / u_read_depth / scale
##        #    s_ = s_obs / s_read_depth
##
##        std_u = u_scale / scale
##        tau = tau_inv(u_, s_, u0, s0, alpha, beta, gamma)
##        ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
##        if self.cell_specific_kinetics is None:
##            tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
##            ut_, st_ = mRNA(tau_, u_inf, s_inf, self.zero, beta, gamma)
##        state_on = ((ut - u_)/std_u)**2 + ((st - s_)/s_scale)**2
##        state_zero = ((ut - u0)/std_u)**2 + ((st - s0)/s_scale)**2
##        if self.cell_specific_kinetics is None:
##            state_inf = ((ut_ - u_inf)/std_u)**2 + ((st_ - s_inf)/s_scale)**2
##            state_off = ((ut_ - u_)/std_u)**2 + ((st_ - s_)/s_scale)**2
##            cell_gene_state_logits = torch.stack([state_on, state_zero, state_off, state_inf], dim=-1).argmin(-1)
##        if self.cell_specific_kinetics is None:
##            state = (cell_gene_state_logits > 1) == self.zero
##            t = torch.where(state, tau+t0, tau_+dt_switching+t0)
##        else:
##            t = softplus(tau + t0)
##        cell_time_loc, cell_time_scale = self.time_encoder(t)
##        t = pyro.sample("cell_time", LogNormal(cell_time_loc, torch.sqrt(cell_time_scale)))
##        return t
##
##    def forward(
##        self,
##        u_obs: Optional[torch.Tensor] = None,
##        s_obs: Optional[torch.Tensor] = None,
##        u_log_library: Optional[torch.Tensor] = None,
##        s_log_library: Optional[torch.Tensor] = None,
##        u_log_library_loc: Optional[torch.Tensor] = None,
##        s_log_library_loc: Optional[torch.Tensor] = None,
##        u_log_library_scale: Optional[torch.Tensor] = None,
##        s_log_library_scale: Optional[torch.Tensor] = None,
##        ind_x: Optional[torch.Tensor] = None,
##        cell_state: Optional[torch.Tensor] = None,
##        time_info: Optional[torch.Tensor] = None,
##    ):
##        cell_plate, gene_plate = self.create_plates(u_obs, s_obs, u_log_library, s_log_library, u_log_library_loc, s_log_library_loc, u_log_library_scale, s_log_library_scale, ind_x, cell_state, time_info)
##
##        with gene_plate, poutine.mask(mask=self.include_prior):
##            alpha = self.alpha
##            gamma = self.gamma
##            beta = self.beta
##
##            if self.cell_specific_kinetics is not None:
##                rho, _ = self.multikinetics_encoder(cell_state)
##                alpha = rho * alpha
##                beta = beta * rho
##                gamma = gamma * rho
##            t0 = pyro.sample("t0", Normal(self.zero, self.one))
##            if self.add_offset:
##                u0 = pyro.sample("u_offset", LogNormal(self.zero, self.one))
##                s0 = pyro.sample("s_offset", LogNormal(self.zero, self.one))
##                ##u0 = pyro.sample("u_offset", Gamma(self.one, self.one))
##                ##s0 = pyro.sample("s_offset", Gamma(self.one, self.one))
##            else:
##                s0 = u0 = self.zero
##
##            if self.likelihood == 'Normal':
##                u_scale = self.u_scale
##                s_scale = self.s_scale
##            else:
##                # NegativeBinomial and Poisson model
##                u_scale = s_scale = self.one
##
##            if self.cell_specific_kinetics is None:
##                dt_switching = self.dt_switching
##                u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
##                u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
##                s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
##                switching = pyro.deterministic("switching", dt_switching + t0, event_dim=0)
##            else:
##                switching = u_inf = s_inf = None
##
##        with cell_plate:
##            t = pyro.sample("cell_time", LogNormal(self.zero, self.one).mask(self.include_prior))
##
##            # physical time constraint or cytotrace constraint
##            if time_info is not None:
##                physical_time = pyro.sample("physical_time", Bernoulli(logits=t), obs=time_info)
##            ## Gioele's suggestion
##            #alpha = alpha * t
##            #beta = beta * t
##            #gamma = gamma * t
##            #dt_switching = dt_switching * t
##            #with gene_plate:
##            #    if self.cell_specific_kinetics is None:
##            #        u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
##            #        u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
##            #        s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
##            #        switching = pyro.deterministic("switching", dt_switching + t0, event_dim=0)
##            #    else:
##            #        switching = u_inf = s_inf = None
##
##        with cell_plate:
##            if self.correct_library_size and (self.likelihood != 'Normal'):
##                if self.correct_library_size == 'cell_size_regress':
##                    #u_read_depth = u_log_library
##                    #s_read_depth = s_log_library
##                    u_read_depth = torch.log(pyro.sample("u_read_depth", LogNormal(u_log_library, u_log_library_scale)))
##                    s_read_depth = torch.log(pyro.sample("s_read_depth", LogNormal(s_log_library, s_log_library_scale)))
##                    # cell-wise coef per cell
##                    u_cell_size_coef = pyro.sample("u_cell_size_coef", Normal(self.zero, self.one))
##                    ut_coef = pyro.sample("ut_coef", Normal(self.zero, self.one))
##                    s_cell_size_coef = pyro.sample("s_cell_size_coef", Normal(self.zero, self.one))
##                    st_coef = pyro.sample("st_coef", Normal(self.zero, self.one))
##                else:
##                    #u_read_depth = torch.exp(u_log_library)
##                    #s_read_depth = torch.exp(s_log_library)
##                    u_read_depth = pyro.sample("u_read_depth", LogNormal(u_log_library, u_log_library_scale))
##                    s_read_depth = pyro.sample("s_read_depth", LogNormal(s_log_library, s_log_library_scale))
##                    u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
##            else:
##                u_read_depth = s_read_depth = None
##                u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
##            with gene_plate:
##                if self.guide_type == 'auto_t0_constraint' or self.guide_type == 'velocity_auto_t0_constraint':
##                    pyro.sample("time_constraint", Bernoulli(logits=t-t0), obs=self.one)
##
##                # constraint u_inf > u0, s_inf > s0, reduce performance..
##                # pyro.sample("u_inf_constraint", Bernoulli(logits=alpha/beta-u0), obs=self.one)
##                # pyro.sample("s_inf_constraint", Bernoulli(logits=alpha/gamma-s0), obs=self.one)
##                # pyro.sample("u_inf_constraint2", Bernoulli(logits=alpha/beta-u_inf), obs=self.one)
##                # pyro.sample("s_inf_constraint2", Bernoulli(logits=alpha/gamma-s_inf), obs=self.one)
##
##                ut, st = self.get_rna(u_scale, s_scale, alpha, beta, gamma,
##                                      t, u0, s0, t0, switching, u_inf, s_inf)
##                u_dist, s_dist = self.get_likelihood(ut, st, u_log_library, s_log_library, u_scale, s_scale, u_read_depth=u_read_depth, s_read_depth=s_read_depth, u_cell_size_coef=u_cell_size_coef, ut_coef=ut_coef, s_cell_size_coef=s_cell_size_coef, st_coef=st_coef)
##                u = pyro.sample("u", u_dist, obs=u_obs)
##                s = pyro.sample("s", s_dist, obs=s_obs)
##        return u, s


class VelocityModelAuto(AuxCellVelocityModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.guide_type in [
            "velocity_auto",
            "velocity_auto_depth",
            "velocity_auto_t0_constraint",
        ]:
            self.time_encoder = TimeEncoder2(
                self.num_genes,
                n_output=1,
                dropout_rate=0.5,
                activation_fn=nn.ELU,
                n_layers=3,
                var_eps=1e-6,
            )
        if self.correct_library_size and (
            self.guide_type == "velocity_auto"
            or self.guide_type == "velocity_auto_t0_constraint"
        ):
            self.u_lib_encoder = TimeEncoder2(
                self.num_genes + 1, 1, n_layers=3, dropout_rate=0.5
            )
            self.s_lib_encoder = TimeEncoder2(
                self.num_genes + 1, 1, n_layers=3, dropout_rate=0.5
            )

        if self.cell_specific_kinetics is not None:
            self.multikinetics_encoder = TimeEncoder2(
                1,  # encode cell state
                1,  # encode cell specificity of kinetics
                dropout_rate=0.5,
                last_layer_activation=nn.Sigmoid(),
                n_layers=3,
            )

    def get_rna(
        self,
        u_scale: torch.Tensor,
        s_scale: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
        s0: torch.Tensor,
        t0: torch.Tensor,
        switching: Optional[torch.Tensor] = None,
        u_inf: Optional[torch.Tensor] = None,
        s_inf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cell_specific_kinetics is None:
            if self.guide_type == "auto":
                enum = "parallel"
            else:
                if pyro.__version__.startswith(
                    "1.8.1"
                ):  # parallel still memory leaky from pip install
                    enum = "parallel"
                elif pyro.__version__.startswith("1.6.0"):
                    # neural network guide only works in sequential enumeration
                    # only 1.6.0 version supports model-side sequential enumeration
                    enum = "sequential"

            state = (
                pyro.sample(
                    "cell_gene_state",
                    Bernoulli(logits=t - switching),
                    infer={"enumerate": enum},
                )
                == self.zero
            )
            alpha_off = self.zero
            u0_vec = torch.where(state, u0, u_inf)
            s0_vec = torch.where(state, s0, s_inf)
            alpha_vec = torch.where(state, alpha, alpha_off)
            tau = softplus(torch.where(state, t - t0, t - switching))
        else:
            u0_vec = u0
            s0_vec = s0
            alpha_vec = alpha
            tau = softplus(t - t0)
            ##tau = relu(torch.where(state, t - t0, t - switching))
        # print(alpha_vec.shape)
        ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
        ut = ut * u_scale / s_scale
        return ut, st

    def get_time(
        self,
        u_scale,
        s_scale,
        alpha,
        beta,
        gamma,
        u_obs,
        s_obs,
        u0,
        s0,
        t0,
        dt_switching,
        u_inf,
        s_inf,
        u_read_depth=None,
        s_read_depth=None,
    ):
        scale = u_scale / s_scale

        # if u_read_depth is None:
        #    u_ = u_obs / scale
        #    s_ = s_obs
        # else:
        #    #neural network correction of read depth not converge
        #    u_ = u_obs / u_read_depth / scale
        #    s_ = s_obs / s_read_depth
        u_ = u_obs / scale
        s_ = s_obs

        std_u = u_scale / scale
        tau = tau_inv(u_, s_, u0, s0, alpha, beta, gamma)
        ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
        if self.cell_specific_kinetics is None:
            tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
            ut_, st_ = mRNA(tau_, u_inf, s_inf, self.zero, beta, gamma)
        state_on = ((ut - u_) / std_u) ** 2 + ((st - s_) / s_scale) ** 2
        state_zero = ((ut - u0) / std_u) ** 2 + ((st - s0) / s_scale) ** 2
        if self.cell_specific_kinetics is None:
            state_inf = ((ut_ - u_inf) / std_u) ** 2 + (
                (st_ - s_inf) / s_scale
            ) ** 2
            state_off = ((ut_ - u_) / std_u) ** 2 + ((st_ - s_) / s_scale) ** 2
            cell_gene_state_logits = torch.stack(
                [state_on, state_zero, state_off, state_inf], dim=-1
            ).argmin(-1)
        if self.cell_specific_kinetics is None:
            state = (cell_gene_state_logits > 1) == self.zero
            t = torch.where(state, tau + t0, tau_ + dt_switching + t0)
        else:
            t = softplus(tau + t0)
        cell_time_loc, cell_time_scale = self.time_encoder(t)
        t = pyro.sample(
            "cell_time", LogNormal(cell_time_loc, torch.sqrt(cell_time_scale))
        )
        return t

    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        u_log_library_loc: Optional[torch.Tensor] = None,
        s_log_library_loc: Optional[torch.Tensor] = None,
        u_log_library_scale: Optional[torch.Tensor] = None,
        s_log_library_scale: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_plate, gene_plate = self.create_plates(
            u_obs,
            s_obs,
            u_log_library,
            s_log_library,
            u_log_library_loc,
            s_log_library_loc,
            u_log_library_scale,
            s_log_library_scale,
            ind_x,
            cell_state,
            time_info,
        )

        with gene_plate, poutine.mask(mask=self.include_prior):
            alpha = self.alpha
            gamma = self.gamma
            beta = self.beta

            # if self.cell_specific_kinetics is not None:
            #     rho, _ = self.multikinetics_encoder(cell_state)
            #     alpha = rho * alpha
            #     beta = beta * rho
            #     gamma = gamma * rho

            if self.add_offset:
                u0 = pyro.sample("u_offset", LogNormal(self.zero, self.one))
                s0 = pyro.sample("s_offset", LogNormal(self.zero, self.one))
            else:
                s0 = u0 = self.zero

            t0 = pyro.sample("t0", Normal(self.zero, self.one))

            if (self.likelihood == "Normal") or (self.guide_type == "auto"):
                u_scale = self.u_scale
                s_scale = self.one
                if self.likelihood == "Normal":
                    s_scale = self.s_scale
            else:
                # NegativeBinomial and Poisson model
                u_scale = s_scale = self.one

            if self.cell_specific_kinetics is None:
                dt_switching = self.dt_switching
                u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
                u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
                s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
                switching = pyro.deterministic(
                    "switching", dt_switching + t0, event_dim=0
                )
            else:
                switching = u_inf = s_inf = None

        with cell_plate:
            t = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )
            # physical time constraint or cytotrace constraint
            # if time_info is not None:
            #     physical_time = pyro.sample("physical_time", Bernoulli(logits=t), obs=time_info)
            ## Gioele's suggestion
            # alpha = alpha * t
            # beta = beta * t
            # gamma = gamma * t
            # dt_switching = dt_switching * t
            # with gene_plate:
            #    if self.cell_specific_kinetics is None:
            #        u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
            #        u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
            #        s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
            #        switching = pyro.deterministic("switching", dt_switching + t0, event_dim=0)
            #    else:
            #        switching = u_inf = s_inf = None

        with cell_plate:
            u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
            u_read_depth = s_read_depth = None
            if self.correct_library_size and (self.likelihood != "Normal"):
                if self.guide_type == "velocity_auto":
                    u_read_depth = torch.exp(u_log_library)
                    s_read_depth = torch.exp(s_log_library)
                else:
                    u_read_depth = pyro.sample(
                        "u_read_depth",
                        LogNormal(u_log_library, u_log_library_scale),
                    )
                    s_read_depth = pyro.sample(
                        "s_read_depth",
                        LogNormal(s_log_library, s_log_library_scale),
                    )
                    if self.correct_library_size == "cell_size_regress":
                        # cell-wise coef per cell
                        u_cell_size_coef = pyro.sample(
                            "u_cell_size_coef", Normal(self.zero, self.one)
                        )
                        ut_coef = pyro.sample(
                            "ut_coef", Normal(self.zero, self.one)
                        )
                        s_cell_size_coef = pyro.sample(
                            "s_cell_size_coef", Normal(self.zero, self.one)
                        )
                        st_coef = pyro.sample(
                            "st_coef", Normal(self.zero, self.one)
                        )
            with gene_plate:
                if (
                    self.guide_type == "auto_t0_constraint"
                    or self.guide_type == "velocity_auto_t0_constraint"
                ):
                    pyro.sample(
                        "time_constraint",
                        Bernoulli(logits=t - t0),
                        obs=self.one,
                    )
                # constraint u_inf > u0, s_inf > s0, reduce performance..
                # pyro.sample("u_inf_constraint", Bernoulli(logits=alpha/beta-u0), obs=self.one)
                # pyro.sample("s_inf_constraint", Bernoulli(logits=alpha/gamma-s0), obs=self.one)
                # pyro.sample("u_inf_constraint2", Bernoulli(logits=alpha/beta-u_inf), obs=self.one)
                # pyro.sample("s_inf_constraint2", Bernoulli(logits=alpha/gamma-s_inf), obs=self.one)
                ut, st = self.get_rna(
                    u_scale,
                    s_scale,
                    alpha,
                    beta,
                    gamma,
                    t,
                    u0,
                    s0,
                    t0,
                    switching,
                    u_inf,
                    s_inf,
                )
                u_dist, s_dist = self.get_likelihood(
                    ut,
                    st,
                    u_log_library,
                    s_log_library,
                    u_scale,
                    s_scale,
                    u_read_depth=u_read_depth,
                    s_read_depth=s_read_depth,
                    u_cell_size_coef=u_cell_size_coef,
                    ut_coef=ut_coef,
                    s_cell_size_coef=s_cell_size_coef,
                    st_coef=st_coef,
                )
                u = pyro.sample("u", u_dist, obs=u_obs)
                s = pyro.sample("s", s_dist, obs=s_obs)
        return u, s
