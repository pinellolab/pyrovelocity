from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import pyro
import pyro.poutine as poutine
import torch
from pyro.distributions import Bernoulli
from pyro.distributions import Beta
from pyro.distributions import LogNormal
from pyro.distributions import NegativeBinomial
from pyro.distributions import Normal
from pyro.distributions import Poisson
from pyro.distributions.constraints import positive
from pyro.nn import PyroModule
from pyro.nn import PyroParam
from pyro.nn import PyroSample
from pyro.primitives import plate
from scvi.nn import Decoder
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
        cell_plate = pyro.plate("cells", self.num_cells, subsample=ind_x, dim=-2)
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
        return pyro.plate("cells", self.num_cells, subsample=ind_x, dim=-1)

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
                )
            else:
                return (
                    Normal(self.zero, self.one * 0.1)
                    .expand((self.num_genes,))
                    .to_event(1)
                    .mask(self.include_prior)
                )
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one).mask(
                self.include_prior
            )
        return (
            LogNormal(self.zero, self.one).expand((self.num_genes,)).to_event(1)
        )

    @PyroSample
    def cell_time(self):
        if self.plate_size == 2 and self.shared_time:
            return Normal(self.zero, self.one)
        else:
            return LogNormal(self.zero, self.one)

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
        cell_state = tensor_dict.get("pyro_cell_state")
        time_info = tensor_dict.get("time_info")
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
        if self.likelihood == "NB":
            if self.correct_library_size:
                ut = relu(ut) + self.one * 1e-6
                st = relu(st) + self.one * 1e-6
                ut = pyro.deterministic("ut", ut, event_dim=0)
                st = pyro.deterministic("st", st, event_dim=0)
                if self.guide_type not in [
                    "velocity_auto",
                    "velocity_auto_depth",
                ]:
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
            u_dist = NegativeBinomial(total_count=self.u_px_r.exp(), logits=u_logits)
            s_dist = NegativeBinomial(total_count=self.s_px_r.exp(), logits=s_logits)
        elif self.likelihood == "Poisson":
            if self.correct_library_size:
                ut = relu(ut) + self.one * 1e-6
                st = relu(st) + self.one * 1e-6
                ut = pyro.deterministic("ut", ut, event_dim=0)
                st = pyro.deterministic("st", st, event_dim=0)
                if self.correct_library_size == "cell_size_regress":
                    ut_sum = torch.log(torch.sum(ut, dim=-1, keepdim=True))
                    st_sum = torch.log(torch.sum(st, dim=-1, keepdim=True))
                    ut = torch.log(ut)
                    st = torch.log(st)
                    ut = torch.exp(
                        ut_coef * ut + u_cell_size_coef * (-ut_sum + u_read_depth)
                    )
                    st = torch.exp(
                        st_coef * st + s_cell_size_coef * (-st_sum + s_read_depth)
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
                )
                s_dist = Normal(st, s_scale)
            else:
                u_dist = Normal(
                    ut, self.one * 0.1
                )
                s_dist = Normal(st, self.one * 0.1)
        elif self.likelihood == "LogNormal":
            if u_scale is not None and s_scale is not None:
                u_dist = LogNormal(
                    (ut + self.one * 1e-6).log(), u_scale
                )
                s_dist = LogNormal((st + self.one * 1e-6).log(), s_scale)
            else:
                u_dist = LogNormal(
                    ut, self.one * 0.1
                )
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
            u_scale = self.u_scale
            s_scale = self.s_scale
            if self.t_scale_on and self.shared_time:
                t_scale = self.t_scale
            if self.latent_factor_operation == "selection":
                p_velocity = self.p_velocity

            if self.latent_factor == "linear":
                u_pcs_mean = pyro.sample("u_pcs_mean", Normal(self.zero, self.one))
                s_pcs_mean = pyro.sample("s_pcs_mean", Normal(self.zero, self.one))
            u_inf, s_inf = mRNA(switching, self.zero, self.zero, alpha, beta, gamma)

        if self.latent_factor == "linear":
            cell_codebook = self.cell_codebook
            with cell_plate, poutine.mask(mask=False):
                cell_code = self.cell_code
        if self.shared_time:
            with cell_plate:
                cell_time = self.cell_time

        u_read_depth = None
        s_read_depth = None

        with cell_plate, gene_plate, poutine.mask(
            mask=pyro.subsample(self.mask.to(alpha.device), event_dim=0)
        ):
            t = self.latent_time
            if self.shared_time:
                t = cell_time * t_scale + t if self.t_scale_on else cell_time + t
            state = (
                pyro.sample("cell_gene_state", Bernoulli(logits=t - switching))
                == self.zero
            )
            u0_vec = torch.where(state, self.zero, u_inf)
            s0_vec = torch.where(state, self.zero, s_inf)
            alpha_vec = torch.where(state, alpha, self.zero)
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
                    regressor_output[..., : self.num_genes].squeeze() + u_pcs_mean
                )
                regressor_s = softplus(
                    regressor_output[..., self.num_genes :].squeeze() + s_pcs_mean
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
        with (
            cell_plate
        ):
            t = self.latent_time
            u_inf, s_inf = mRNA(switching, self.zero, self.zero, alpha, beta, gamma)
            state = (
                pyro.sample(
                    "cell_gene_state", Bernoulli(logits=t - switching).to_event(1)
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
            return self.model2(u_obs, s_obs, u_log_library, s_log_library, ind_x)


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

            gene_offset = self.gene_offset if self.add_offset else self.zero
            switching = dt_switching + gene_offset
            # u_inf, s_inf = mRNA(switching, self.zero, self.zero, alpha, beta, gamma)
            # u_inf, s_inf = mRNA(dt_switching, self.zero, self.zero, alpha, beta, gamma)
            u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)

            # u0 = torch.where(u0 > u_inf, u_inf, u0)
            # s0 = torch.where(s0 > s_inf, s_inf, s0)

            if self.latent_factor_operation == "selection":
                p_velocity = pyro.sample("p_velocity", Beta(self.one * 5, self.one))
            else:
                p_velocity = None

            if self.latent_factor == "linear":
                u_pcs_mean = pyro.sample("u_pcs_mean", Normal(self.zero, self.one))
                s_pcs_mean = pyro.sample("s_pcs_mean", Normal(self.zero, self.one))
            else:
                u_pcs_mean, s_pcs_mean = None, None

        s_read_depth = None
        # with cell_plate:
        #    u_read_depth = pyro.sample('u_read_depth', dist.LogNormal(u_log_library, self.one))
        #    s_read_depth = pyro.sample('s_read_depth', dist.LogNormal(s_log_library, self.one))
        u_read_depth = None
        # same as before, just refactored slightly
        # with cell_plate, gene_plate:
        #     cell_gene_mask = pyro.subsample(self.mask.to(alpha.device), event_dim=0)

        cell_codebook = self.cell_codebook if self.latent_factor == "linear" else None
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
                    aux_u_log_library = torch.log(aux_u_obs.sum(axis=-1))  # TODO define
                    aux_s_log_library = torch.log(aux_s_obs.sum(axis=-1))  # TODO define
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
                "cell_time", LogNormal(self.zero, self.one).mask(self.include_prior)
            )  # mask=False works for cpm and raw read count, count needs steady-state initialization
        ##assert cell_time.shape == (256, 1)

        with gene_plate:
            if self.latent_factor_operation == "selection":
                cellgene_type = pyro.sample("cellgene_type", Bernoulli(p_velocity))
            else:
                cellgene_type = None

            if self.only_cell_times:
                if self.decoder_on:
                    t, _ = self.decoder(cell_time)
                else:
                    t = cell_time

            elif self.shared_time:
                t = pyro.sample(
                    "latent_time",
                    Normal(self.zero, self.one).mask(self.include_prior),
                )
                t = (
                    cell_time * t_scale + t + gene_offset
                    if self.t_scale_on and t_scale is not None
                    else cell_time + t
                )
            else:
                t = pyro.sample(
                    "latent_time",
                    LogNormal(self.zero, self.one).mask(self.include_prior),
                )
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
                    regressor_output[..., : self.num_genes].squeeze() + u_pcs_mean
                )
                regressor_s = softplus(
                    regressor_output[..., self.num_genes :].squeeze() + s_pcs_mean
                )
            if self.latent_factor_operation == "selection":
                ut = torch.where(
                    cellgene_type == self.one,
                    ut * u_scale / s_scale,
                    softplus(regressor_u),
                )
                st = torch.where(cellgene_type == self.one, st, softplus(regressor_s))
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


class VelocityModelAuto(AuxCellVelocityModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # if self.guide_type in [
        #     "velocity_auto",
        #     "velocity_auto_depth",
        #     "velocity_auto_t0_constraint",
        # ]:
        #     self.time_encoder = TimeEncoder2(
        #         self.num_genes,
        #         n_output=1,
        #         dropout_rate=0.5,
        #         activation_fn=nn.ELU,
        #         n_layers=3,
        #         var_eps=1e-6,
        #     )
        # if self.correct_library_size and (
        #     self.guide_type == "velocity_auto"
        #     or self.guide_type == "velocity_auto_t0_constraint"
        # ):
        #     self.u_lib_encoder = TimeEncoder2(
        #         self.num_genes + 1, 1, n_layers=3, dropout_rate=0.5
        #     )
        #     self.s_lib_encoder = TimeEncoder2(
        #         self.num_genes + 1, 1, n_layers=3, dropout_rate=0.5
        #     )

        # if self.cell_specific_kinetics is not None:
        #     self.multikinetics_encoder = TimeEncoder2(
        #         1,  # encode cell state
        #         1,  # encode cell specificity of kinetics
        #         dropout_rate=0.5,
        #         last_layer_activation=nn.Sigmoid(),
        #         n_layers=3,
        #     )

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
            if (
                self.guide_type != "auto"
                and pyro.__version__.startswith("1.8.1")
                or self.guide_type == "auto"
            ):  # parallel leads to memory leak
                enum = "parallel"
            elif (
                self.guide_type != "auto"
                and not pyro.__version__.startswith("1.8.1")
                and pyro.__version__.startswith("1.6.0")
            ):
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
            state_inf = ((ut_ - u_inf) / std_u) ** 2 + ((st_ - s_inf) / s_scale) ** 2
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
            else:
                # NegativeBinomial and Poisson model
                u_scale = s_scale = self.one

            if self.likelihood == "Normal":
                s_scale = self.s_scale
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
                "cell_time", LogNormal(self.zero, self.one).mask(self.include_prior)
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
                        "u_read_depth", LogNormal(u_log_library, u_log_library_scale)
                    )
                    s_read_depth = pyro.sample(
                        "s_read_depth", LogNormal(s_log_library, s_log_library_scale)
                    )
                    if self.correct_library_size == "cell_size_regress":
                        # cell-wise coef per cell
                        u_cell_size_coef = pyro.sample(
                            "u_cell_size_coef", Normal(self.zero, self.one)
                        )
                        ut_coef = pyro.sample("ut_coef", Normal(self.zero, self.one))
                        s_cell_size_coef = pyro.sample(
                            "s_cell_size_coef", Normal(self.zero, self.one)
                        )
                        st_coef = pyro.sample("st_coef", Normal(self.zero, self.one))
            with gene_plate:
                if self.guide_type in [
                    "auto_t0_constraint",
                    "velocity_auto_t0_constraint",
                ]:
                    pyro.sample(
                        "time_constraint", Bernoulli(logits=t - t0), obs=self.one
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
