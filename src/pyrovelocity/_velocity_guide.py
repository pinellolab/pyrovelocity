from typing import Iterable, Optional

import pyro
import torch
from pyro import poutine
from pyro.contrib import easyguide
from pyro.distributions import (
    Bernoulli,
    Categorical,
    Delta,
    Normal,
)
from pyro.distributions.constraints import positive
from pyro.infer import autoguide
from pyro.infer.autoguide.guides import AutoGuideList
from pyro.nn import PyroModule, PyroParam
from scvi.nn import Encoder, FCLayers
from torch import nn
from torch.nn.functional import softmax, softplus

# from ._velocity_model import LatentFactor
from ._velocity_model import VelocityModel
from .utils import mRNA, site_is_discrete, tau_inv


class VelocityGuide(easyguide.EasyGuide):
    def __init__(
        self,
        model: VelocityModel,
        likelihood: str,
        shared_time: bool,
        t_scale: bool,
        latent_factor: str = "none",
        latent_factor_operation: str = "selection",
        model_type: str = "none",
        plate_size: int = 2,
        only_cell_times: bool = False,
        add_offset: bool = False,
        **initial_values,
    ):
        super().__init__(model)
        self.num_genes = model.num_genes
        self.num_cells = model.num_cells
        self.num_aux_cells = model.num_aux_cells

        self.zero = model.zero
        self.one = model.one
        self.shared_time = shared_time
        self.t_scale_on = t_scale
        self.latent_factor = latent_factor
        self.latent_factor_operation = latent_factor_operation
        self.model_type = model_type
        self.only_cell_times = only_cell_times

        self.plate_size = plate_size
        self.event_dim = 0 if self.model_type in ["velocity", "traj"] else 1
        self.likelihood = likelihood
        self.add_offset = add_offset

        self.mask = initial_values.get(
            "mask", torch.ones(self.num_cells, self.num_genes).bool()
        )
        for key in initial_values:
            self.register_buffer(f"{key}_init", initial_values[key])

        if hasattr(self, "alpha_init") and hasattr(
            self, "u_inf_init"
        ):  # with initialization of steady-state
            self.u_inf = PyroParam(
                self.u_inf_init.clone(),
                constraint=positive,
                event_dim=self.event_dim,
            )
            self.s_inf = PyroParam(
                self.s_inf_init.clone(),
                constraint=positive,
                event_dim=self.event_dim,
            )
        else:
            self.u_inf = PyroParam(
                torch.ones(self.num_genes), constraint=positive, event_dim=0
            )
            self.s_inf = PyroParam(
                torch.ones(self.num_genes), constraint=positive, event_dim=0
            )
        ###if not self.shared_time:
        if not (hasattr(self, "alpha_init") and hasattr(self, "u_inf_init")):
            self.cell_time_init = None
            self.cell_time_aux_init = None

        if self.latent_factor != "linear":
            self.cell_code_init = None
            self.cell_code_aux_init = None

        if self.likelihood != "Normal":
            hidden_size = 128
            self.neuralnetwork = nn.Sequential(
                nn.Linear(self.num_genes * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_genes * 2),
            )

        # self.kinetics_encoder = KineticsParamEncoder(10, self.num_cells)
        # self.genetime_encoder = GeneTimeEncoder(10)

    def init(self, site):
        if hasattr(self, "alpha_init") and hasattr(self, "u_inf_init"):
            if self.num_aux_cells > 0:
                if site["name"] == "aux/cell_code":
                    return autoguide.init_to_value(
                        site, values={site["name"]: self.cell_code_aux_init}
                    )
                if site["name"] == "aux/latent_time":
                    return autoguide.init_to_value(
                        site,
                        values={
                            site["name"]: pyro.subsample(
                                self.latent_time_aux_init, event_dim=0
                            )
                        },
                    )
                if site["name"] == "aux/cell_gene_state":
                    return autoguide.init_to_value(
                        site,
                        values={
                            site["name"]: pyro.subsample(
                                self.cell_gene_state_aux_init, event_dim=0
                            )
                        },
                    )
                if site["name"] == "aux/cell_time":
                    return autoguide.init_to_value(
                        site,
                        values={
                            site["name"]: pyro.subsample(
                                self.cell_time_aux_init, event_dim=0
                            )
                        },
                    )
            if site["name"] == "cell_code":
                return autoguide.init_to_value(
                    site,
                    values={
                        site["name"]: pyro.subsample(
                            self.cell_code_init, event_dim=1
                        )
                    },
                )
            if site["name"] == "cell_codebook":
                return autoguide.init_to_value(
                    site,
                    values={
                        site["name"]: pyro.subsample(
                            self.cell_codebook_init, event_dim=1
                        )
                    },
                )
            if site["name"] == "latent_time":
                if self.num_aux_cells > 0:
                    return autoguide.init_to_value(
                        site,
                        values={
                            site["name"]: pyro.subsample(
                                torch.zeros_like(self.latent_time_init),
                                event_dim=self.event_dim,
                            )
                        },
                    )
                else:
                    return autoguide.init_to_value(
                        site,
                        values={
                            site["name"]: pyro.subsample(
                                self.latent_time_init, event_dim=self.event_dim
                            )
                        },
                    )
            if site["name"] == "cell_gene_state":
                return autoguide.init_to_value(
                    site,
                    values={
                        site["name"]: pyro.subsample(
                            self.cell_gene_state_init, event_dim=self.event_dim
                        )
                    },
                )
            if site["name"] == "cell_time":
                assert getattr(self, f"{site['name']}_init").shape[-1] == 1
                print("guide init-------")
                print(self.cell_time_init.shape)
                print(
                    pyro.subsample(
                        self.cell_time_init, event_dim=self.event_dim
                    ).shape
                )
                print(self.event_dim)
                return autoguide.init_to_value(
                    site,
                    values={
                        site["name"]: pyro.subsample(
                            self.cell_time_init, event_dim=self.event_dim
                        )
                    },
                )
            if site["name"] == "t_scale":
                t_scale_init = self.alpha_init.new_ones(self.num_genes)
                return autoguide.init_to_value(
                    site, values={site["name"]: t_scale_init}
                )
            if site["name"] == "genecellpair_type":
                p_gene = self.alpha_init.new_ones(self.num_genes)
                return autoguide.init_to_value(
                    site, values={site["name"]: p_gene}
                )
            if site["name"] == "p_velocity":
                p_velocity = self.alpha_init.new_ones(self.num_genes)
                return autoguide.init_to_value(
                    site, values={site["name"]: p_velocity}
                )
            if hasattr(self, f"{site['name']}_init"):
                return autoguide.init_to_value(
                    site,
                    values={
                        site["name"]: getattr(self, f"{site['name']}_init")
                    },
                )
        return super().init(site)

    @PyroParam(constraint=positive, event_dim=0)
    def cell_code_scale(self):
        # return self.u_inf_init.new_ones(1)
        return self.u_inf.new_ones(1)

    def guide(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        if self.plate_size == 2:
            cell_plate = self.plate(
                "cells", self.num_cells, subsample=ind_x, dim=-2
            )
            gene_plate = self.plate("genes", self.num_genes, dim=-1)
            if self.latent_factor == "linear":
                decoder_weights = self.map_estimate("cell_codebook")
                encoder_weights = decoder_weights.T

            with gene_plate:
                alpha = self.map_estimate("alpha")
                beta = self.map_estimate("beta")
                gamma = self.map_estimate("gamma")
                u_scale = self.map_estimate("u_scale")
                s_scale = self.map_estimate("s_scale")
                scale = u_scale / s_scale
                u_inf = self.u_inf
                s_inf = self.s_inf
                # u_inf = self.map_estimåte("u_inf")
                # s_inf = self.map_estimate("s_inf")
                # gamma = alpha / s_inf
                # beta = alpha / u_inf
                # pyro.sample("gamma", Delta(gamma))
                # pyro.sample("beta", Delta(beta))
                if self.t_scale_on:
                    t_scale = self.map_estimate("t_scale")
                switching = tau_inv(
                    u_inf, s_inf, self.zero, self.zero, alpha, beta, gamma
                )
                pyro.sample("switching", Delta(switching))

                if self.latent_factor == "linear":
                    u_pcs_mean = self.map_estimate("u_pcs_mean")
                    s_pcs_mean = self.map_estimate("s_pcs_mean")

            if not (self.likelihood in ["Normal", "LogNormal"]):
                # with cell_plate:
                #    u_read_depth = pyro.sample('u_read_depth', LogNormal(u_log_library, self.one.to(u_log_library.device)*0.01))
                #    s_read_depth = pyro.sample('s_read_depth', LogNormal(s_log_library, self.one.to(u_log_library.device)*0.01))
                with cell_plate, gene_plate, poutine.mask(
                    mask=pyro.subsample(
                        self.mask.to(u_obs.device), event_dim=self.event_dim
                    )
                ):
                    # u_s_loc = self.neuralnetwork(torch.cat([u_loc, s_loc], dim=-1))
                    # u_loc, s_loc = u_s_loc[..., :self.num_genes], u_s_loc[..., self.num_genes:]
                    # assert u_loc.shape == (u_obs.shape[0], self.num_genes)
                    # ut = pyro.sample("ut", Delta(torch.exp(u_loc).clamp(max=1e6)))
                    # st = pyro.sample("st", Delta(torch.exp(s_loc).clamp(max=1e6)))
                    ut = u_obs  # / u_read_depth
                    st = s_obs  # / s_read_depth
                    # ut = torch.log1p(u_obs / u_read_depth)
                    # st = torch.log1p(s_obs / s_read_depth)

            if self.shared_time:
                with cell_plate:
                    cell_time = self.map_estimate("cell_time")

            if self.latent_factor == "linear":
                cell_code_scale = pyro.param(
                    "cell_code_scale",
                    lambda: torch.tensor(0.1),
                    constraint=positive,
                ).to(u_obs.device)
                with cell_plate:
                    cell_code_loc = (
                        torch.cat(
                            (u_obs - u_pcs_mean, s_obs - s_pcs_mean), dim=-1
                        )
                        @ encoder_weights
                    )
                    cell_code_loc = cell_code_loc.unsqueeze(-1).transpose(
                        -1, -2
                    )
                    cell_code = pyro.sample(
                        "cell_code",
                        Normal(cell_code_loc, cell_code_scale).to_event(1),
                    )

            with cell_plate, gene_plate, poutine.mask(
                mask=pyro.subsample(self.mask.to(alpha.device), event_dim=0)
            ):
                if self.latent_factor == "linear":
                    regressor_output = torch.einsum(
                        "abc,cd->ad", cell_code, decoder_weights.squeeze()
                    )
                    regressor_u = softplus(
                        regressor_output[..., : self.num_genes].squeeze()
                        + u_pcs_mean
                    )
                    regressor_s = softplus(
                        regressor_output[..., self.num_genes :].squeeze()
                        + s_pcs_mean
                    )
                if self.latent_factor_operation == "sum":
                    u_obs = (u_obs - regressor_u).clamp(0.0)
                    s_obs = (s_obs - regressor_s).clamp(0.0)

                if not (self.likelihood in ["Normal", "LogNormal"]):
                    u_ = ut / scale
                    s_ = st
                else:
                    u_ = u_obs / scale
                    s_ = s_obs

                tau = tau_inv(u_, s_, self.zero, self.zero, alpha, beta, gamma)
                ut, st = mRNA(tau, self.zero, self.zero, alpha, beta, gamma)
                tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
                ut_, st_ = mRNA(tau_, u_inf, s_inf, self.zero, beta, gamma)
                std_u = u_scale / scale
                state_on = ((ut - u_) / std_u) ** 2 + ((st - s_) / s_scale) ** 2
                state_off = ((ut_ - u_) / std_u) ** 2 + (
                    (st_ - s_) / s_scale
                ) ** 2
                cell_gene_state_logits = state_on - state_off
                # debug(cell_gene_state_logits)
                state = (
                    pyro.sample(
                        "cell_gene_state",
                        Bernoulli(logits=cell_gene_state_logits).to_event(
                            self.event_dim
                        ),
                    )
                    == self.zero
                )
                t = torch.where(state, tau, tau_ + switching)
                if self.shared_time:
                    if self.t_scale_on:
                        t = t - cell_time * t_scale
                    else:
                        t = t - cell_time
                pyro.sample("latent_time", Delta(t))

                if self.latent_factor_operation == "selection":
                    # p_velocity = self.map_estimate('p_velocity')
                    ut = torch.where(state, ut, ut_)
                    st = torch.where(state, st, st_)
                    velocity_model = ((ut - u_) / std_u) ** 2 + (
                        (st - s_) / s_scale
                    ) ** 2
                    pca_model = ((regressor_u / scale - u_) / std_u) ** 2 + (
                        (regressor_s - s_) / s_scale
                    ) ** 2
                    p_gene_type = pca_model - velocity_model  # .sum(axis=-2)
                    gene_type = pyro.sample(
                        "genecellpair_type", Bernoulli(logits=p_gene_type)
                    )
        else:
            self.guide2(u_obs, s_obs, u_log_library, s_log_library, ind_x)

    def guide2(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        alpha = self.map_estimate("alpha")
        beta = self.map_estimate("beta")
        gamma = self.map_estimate("gamma")
        u_scale = self.map_estimate("u_scale")
        s_scale = self.map_estimate("s_scale")
        scale = u_scale / s_scale
        u_inf = pyro.param(
            "u_inf",
            lambda: self.u_inf_init.clone(),
            constraint=positive,
            event_dim=self.event_dim,
        ).to(u_obs.device)
        s_inf = pyro.param(
            "s_inf",
            lambda: self.s_inf_init.clone(),
            constraint=positive,
            event_dim=self.event_dim,
        ).to(u_obs.device)
        switching = tau_inv(
            u_inf, s_inf, self.zero, self.zero, alpha, beta, gamma
        )
        pyro.sample("switching", Delta(switching).to_event(1))

        cell_plate = self.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-1
        )
        with cell_plate:  # , poutine.mask(mask=pyro.subsample(self.mask.to(u_obs.device), event_dim=self.event_dim)):
            u_ = u_obs / scale
            s_ = s_obs
            tau = tau_inv(u_, s_, self.zero, self.zero, alpha, beta, gamma)
            ut, st = mRNA(tau, self.zero, self.zero, alpha, beta, gamma)
            tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
            ut_, st_ = mRNA(tau_, u_inf, s_inf, self.zero, beta, gamma)
            std_u = u_scale / scale
            state_on = ((ut - u_) / std_u) ** 2 + ((st - s_) / s_scale) ** 2
            state_off = ((ut_ - u_) / std_u) ** 2 + ((st_ - s_) / s_scale) ** 2
            cell_gene_state_logits = state_on - state_off
            # debug(cell_gene_state_logits)
            state = (
                pyro.sample(
                    "cell_gene_state",
                    Bernoulli(logits=cell_gene_state_logits).to_event(
                        self.event_dim
                    ),
                )
                == self.zero
            )
            t = torch.where(state, tau, tau_ + switching)
            pyro.sample("latent_time", Delta(t).to_event(self.event_dim))


class AuxCellVelocityGuide(VelocityGuide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_cell = CellGuide(
            self.num_cells,
            self.num_genes,
            self.cell_time_init,
            self.zero,
            self.one,
            self.shared_time,
            self.likelihood,
            self.latent_factor,
            self.latent_factor_operation,
            self.t_scale_on,
        )
        if self.num_aux_cells > 0:
            self.infer_aux_cell = CellGuide(
                self.num_aux_cells,
                self.num_genes,
                self.cell_time_aux_init,
                self.zero,
                self.one,
                self.shared_time,
                self.likelihood,
                self.latent_factor,
                self.latent_factor_operation,
                self.t_scale_on,
            )

    def guide(
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
        cell_plate = self.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = self.plate("genes", self.num_genes, dim=-1)

        with gene_plate:
            alpha = self.map_estimate("alpha")
            beta = self.map_estimate("beta")
            gamma = self.map_estimate("gamma")
            u_scale = self.map_estimate("u_scale")
            s_scale = self.map_estimate("s_scale")
            u0 = self.map_estimate("u_offset")
            s0 = self.map_estimate("s_offset")

            # if self.t_scale_on and self.shared_time:
            #    t_scale = self.map_estimate("t_scale")
            # else:
            #    t_scale = None
            #    gene_offset = None
            # alpha, beta, gamma = AutoDiagonalNormal()

            t_scale = None
            # u0 = s0 = 0
            # t_scale = self.map_estimate("t_scale")
            if self.add_offset:
                gene_offset = self.map_estimate("gene_offset")
            else:
                gene_offset = 0

            if self.only_cell_times:
                dt_switching = self.map_estimate("dt_switching")
                u_inf, s_inf = mRNA(
                    dt_switching, u0, s0, self.alpha, self.beta, self.gamma
                )

                ##s_inf/u_inf from alpha/beta/gamma computation
                ##leads all genes to be uniphases, perhaps due to initialization?
                ##with slightly better shared time prediction in dentate gyrus
                # s_inf = alpha / gamma
                # u_inf = alpha / beta
                ##u0 = torch.where(u0 > u_inf, u_inf, u0)
                ##s0 = torch.where(s0 > s_inf, s_inf, s0)
                # dt_switching = tau_inv(u_inf, s_inf, u0, s0, alpha, beta, gamma)
                # dt_switching = pyro.sample("dt_switching", Delta(dt_switching))
            else:
                u_inf = pyro.param(
                    "u_inf",
                    lambda: self.u_inf_init.clone(),
                    constraint=positive,
                    event_dim=self.event_dim,
                ).to(u_obs.device)
                s_inf = pyro.param(
                    "s_inf",
                    lambda: self.s_inf_init.clone(),
                    constraint=positive,
                    event_dim=self.event_dim,
                ).to(u_obs.device)
                switching = tau_inv(
                    u_inf, s_inf, self.zero, self.zero, alpha, beta, gamma
                )
                pyro.sample("switching", Delta(switching))

            if self.latent_factor == "linear":
                u_pcs_mean = self.map_estimate("u_pcs_mean")
                s_pcs_mean = self.map_estimate("s_pcs_mean")
                if self.latent_factor_operation == "selection":
                    p_velocity = self.map_estimate("p_velocity")
                else:
                    p_velocity = None
            else:
                u_pcs_mean = None
                s_pcs_mean = None
                p_velocity = None

        # with cell_plate, gene_plate:
        #    cell_gene_mask = pyro.subsample(self.mask.to(alpha.device), event_dim=0)

        if self.latent_factor == "linear":
            decoder_weights = self.map_estimate("cell_codebook")
        else:
            decoder_weights = None

        # same as before, just refactored a bit
        with cell_plate:  # , poutine.mask(mask=cell_gene_mask):
            u_read_depth = None
            s_read_depth = None
            ut, st, u_, s_, regressor_u, regressor_s = self.infer_cell(
                gene_plate,
                u_obs,
                s_obs,
                u_log_library,
                s_log_library,
                alpha,
                beta,
                gamma,
                # u_inf, s_inf, switching, u_scale, s_scale,
                u_inf,
                s_inf,
                dt_switching,
                u_scale,
                s_scale,
                u_pcs_mean,
                s_pcs_mean,
                t_scale,
                decoder_weights,
                gene_offset,
                p_velocity,
                self.only_cell_times,
                u_read_depth=u_read_depth,
                s_read_depth=s_read_depth,
                u0=u0,
                s0=s0,
            )  # TODO arguments and return type
        if self.num_aux_cells > 0:
            # new: infer latent variables for auxiliary cells
            with pyro.contrib.autoname.scope(prefix="aux"):
                # TODO set self.num_aux_cells
                aux_cell_plate = self.plate(
                    "aux_cell_plate", self.num_aux_cells, dim=cell_plate.dim
                )
                with aux_cell_plate:
                    aux_u_obs = pyro.param(
                        "aux_u_obs",
                        lambda: self.aux_u_obs_init,
                        constraint=positive,
                        event_dim=0,
                    )  # TODO initialize
                    aux_s_obs = pyro.param(
                        "aux_s_obs",
                        lambda: self.aux_s_obs_init,
                        constraint=positive,
                        event_dim=0,
                    )  # TODO initialize
                    aux_u_log_library = torch.log(
                        aux_u_obs.sum(axis=-1) + 1e-6
                    )  # TODO define
                    aux_s_log_library = torch.log(
                        aux_s_obs.sum(axis=-1) + 1e-6
                    )  # TODO define
                    (
                        aux_ut,
                        aux_st,
                        aux_u_,
                        aux_s_,
                        aux_regressor_u,
                        aux_regressor_s,
                    ) = self.infer_aux_cell(
                        gene_plate,
                        aux_u_obs,
                        aux_s_obs,
                        aux_u_log_library,
                        aux_s_log_library,
                        alpha,
                        beta,
                        gamma,
                        # u_inf, s_inf, switching, u_scale, s_scale,
                        u_inf,
                        s_inf,
                        dt_switching,
                        u_scale,
                        s_scale,
                        u_pcs_mean,
                        s_pcs_mean,
                        t_scale,
                        decoder_weights,
                        gene_offset,
                        p_velocity,
                        self.only_cell_times,
                        u_read_depth=u_read_depth,
                        s_read_depth=s_read_depth,
                        u0=u0,
                        s0=s0,
                    )  # TODO arguments and return type
            # if self.latent_factor_operation == 'selection':
            #    ut = torch.cat([ut, aux_ut], dim=cell_plate.dim)
            #    st = torch.cat([st, aux_st], dim=cell_plate.dim)
            #    u_ = torch.cat([u_, aux_u_], dim=cell_plate.dim)
            #    s_ = torch.cat([s_, aux_s_], dim=cell_plate.dim)
            #    regressor_u = torch.cat([regressor_u, aux_regressor_u], dim=cell_plate.dim)
            #    regressor_s = torch.cat([regressor_s, aux_regressor_s], dim=cell_plate.dim)

        # gene selection version
        # cell-gene selection should be moved into infer_cell
        # if self.latent_factor_operation == 'selection':
        #    with gene_plate, self.plate("cells", self.num_cells+self.num_aux_cells, subsample=ind_x, dim=-2):
        #        scale = u_scale / s_scale
        #        std_u = u_scale / scale
        #        velocity_error = ((ut - u_)/std_u) ** 2 + ((st - s_)/s_scale) ** 2
        #        pca_error = ((regressor_u/scale - u_)/std_u) ** 2 + ((regressor_s - s_)/s_scale) ** 2
        #        probs = torch.stack([
        #            torch.exp(-velocity_error) - p_velocity,
        #            torch.exp(-pca_error) - (1 - p_velocity)
        #        ], dim=-1)
        #        probs = softmax(probs, dim=-1)  # probs.sum(-1, keepdim=True)
        #        p_gene_type = probs[..., 0]
        #        gene_type = pyro.sample("cellgene_type", Bernoulli(p_gene_type)) ## tend to be constant


class MultiKineticsGuide(VelocityGuide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.latent_kinetics_comp = 10
        ##option 1: encoder for latent space of alpha/beta/gamma
        # self.us_encoder = Encoder(
        #    self.num_genes * 2,
        #    n_output=self.latent_kinetics_comp,
        #    activation_fn=nn.ELU,
        #    n_layers=2, var_eps=1e-6)
        ##option 1: decoder of alpha/beta/gamma
        # self.infer_kinetics = Decoder(
        #    self.latent_kinetics_comp,
        #    self.num_genes*3, # * 6,
        #    n_layers=2,
        #    activation_fn=nn.ELU,
        #    n_hidden=128,
        # )
        self.time_encoder = TimeEncoder(
            self.num_genes,
            n_output=1,
            activation_fn=nn.ELU,
            n_layers=2,
            var_eps=1e-6,
        )
        self.k = 2

    def guide(
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
        # pyro.module("gamma_celltime_encoder", self.gamma_celltime_encoder)
        # pyro.module("infer_kinetics", self.infer_kinetics)
        # pyro.module("us_encoder", self.us_encoder)
        pyro.module("time_encoder", self.time_encoder)
        cell_plate = self.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = self.plate("genes", self.num_genes, dim=-1)
        kinetics_plate = self.plate("kinetics", self.k, dim=-3)
        with kinetics_plate, gene_plate:
            alpha_k = self.map_estimate("alpha")
            beta_k = self.map_estimate("beta")
            gamma_k = self.map_estimate("gamma")
            dt_switching_k = self.map_estimate("dt_switching")
            # t0_k = self.map_estimate('t0')
            # u_offset_k = self.map_estimate('u_offset')
            # s_offset_k = self.map_estimate('s_offset')
            # u_scale = self.map_estimate('u_scale')
            # s_scale = self.map_estimate('s_scale')

        with gene_plate:
            u0 = s0 = self.zero
            u_scale = self.map_estimate("u_scale")
            s_scale = self.map_estimate("s_scale")
            t0 = self.map_estimate("t0")
            u0 = u_offset = self.map_estimate("u_offset")
            s0 = s_offset = self.map_estimate("s_offset")

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

        # dt_switching = dt_switching_k[cluster_ind].squeeze()
        # u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
        # switching = t0 + dt_switching

        # with cell_plate:
        #    latent_kinetics_loc, latent_kinetics_scale, z = self.us_encoder(torch.hstack([u_obs / (self.u_scale / self.s_scale), s_obs]))
        # kinetics_params, _ = self.infer_kinetics(latent_kinetics_loc.squeeze())
        # alpha = softplus(kinetics_params[:, :self.num_genes])
        # beta = softplus(kinetics_params[:, self.num_genes:(self.num_genes*2)])
        # gamma = softplus(kinetics_params[:, (self.num_genes*2):(self.num_genes*3)])
        ##t0 = kinetics_params[:, (self.num_genes*3):(self.num_genes*4)]
        ##u0 = softplus(kinetics_params[:, (self.num_genes*4):(self.num_genes*5)])
        ##s0 = softplus(kinetics_params[:, (self.num_genes*5):(self.num_genes*6)])
        # with cell_plate, gene_plate:
        #    pyro.sample("alpha", Delta(alpha, event_dim=0))
        #    pyro.sample("beta", Delta(beta, event_dim=0))
        #    pyro.sample("gamma", Delta(gamma, event_dim=0))
        # pyro.sample("t0", Delta(t0, event_dim=0))
        # pyro.sample("u0", Delta(u0, event_dim=0))
        # pyro.sample("s0", Delta(s0, event_dim=0))

        scale = u_scale / s_scale
        s_ = s_obs
        u_ = u_obs / scale

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
        t = torch.where(state, tau + t0, tau_ + switching)
        ##t = tau + t0
        cell_time_loc = self.time_encoder(t)
        with cell_plate:
            pyro.sample("cell_time", Delta(cell_time_loc, event_dim=0))


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
        # self.var_encoder = nn.Linear(n_hidden, n_output)
        # if distribution == "ln":
        #    self.z_transformation = nn.Softmax(dim=-1)
        # else:
        #    self.z_transformation = identity

    # def forward(self, x: torch.Tensor, *cat_list: int, u_log_library: torch.Tensor, s_log_library: torch.Tensor):
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
        # q_m = self.mean_encoder(q) # really big cell_time, this also generates reverse order with poisson
        # q1 = self.act(q_m[:, 0].reshape((-1, 1)))
        # q2 = q_m[:, 1].reshape((-1, 1))
        # q3 = self.act(q_m[:, 2].reshape((-1, 1)))
        # q4 = q_m[:, 3].reshape((-1, 1))
        # q5 = self.act(q_m[:, 4].reshape((-1, 1)))
        # q_m = self.mean_encoder(q)           # huge cell_time, this generates reverse order
        # q_m = torch.tanh(self.mean_encoder(q)) # contrain not work
        # q_v = torch.exp(self.var_encoder(q)) + self.var_eps # This pretty poor performance
        # q_v = nn.ReLU()(self.var_encoder(q)) + self.var_eps # nan values
        # q_v = self.act(self.var_encoder(q)) + self.var_eps  # slightly better..
        # latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m  # , q_v
        # return q1, q2, q3, q4, q5


class CellGuide(PyroModule):
    def __init__(
        self,
        num_cells,
        num_genes,
        init_cell_time,
        zero,
        one,
        shared_time,
        likelihood,
        latent_factor,
        latent_factor_operation,
        t_scale_on,
    ):
        super().__init__()
        self.num_cells = num_cells  # TODO this shouldn't be required
        self.num_genes = num_genes
        self.shared_time = shared_time
        self.latent_factor = latent_factor
        self.latent_factor_operation = latent_factor_operation
        self.t_scale_on = t_scale_on
        if self.shared_time:
            if not (init_cell_time is None):
                init_cell_time = (
                    init_cell_time.detach().clone().requires_grad_()
                )
                self.cell_time_loc = PyroParam(init_cell_time, event_dim=0)
        self.cell_code_scale = PyroParam(
            torch.tensor(0.1), constraint=positive, event_dim=0
        )
        self.zero = zero
        self.one = one
        self.u_loc = PyroParam(
            torch.ones(self.num_cells, self.num_genes),
            constraint=positive,
            event_dim=0,
        )
        self.s_loc = PyroParam(
            torch.ones(self.num_cells, self.num_genes),
            constraint=positive,
            event_dim=0,
        )
        self.likelihood = likelihood
        self.encoder = TimeEncoder(
            self.num_genes,  # +2 for U and S library sizes
            n_output=1,
            dropout_rate=0.5,
            activation_fn=nn.ELU,
            n_layers=3,
            var_eps=1e-6,
        )

    def forward(
        self,
        gene_plate,
        u_obs,
        s_obs,
        u_log_library,
        s_log_library,
        alpha,
        beta,
        gamma,
        # u_inf, s_inf, switching, u_scale, s_scale,
        u_inf,
        s_inf,
        dt_switching,
        u_scale,
        s_scale,
        u_pcs_mean,
        s_pcs_mean,
        t_scale,
        decoder_weights,
        gene_offset,
        p_velocity=None,
        only_cell_times=False,
        u_read_depth=None,
        s_read_depth=None,
        u0=None,
        s0=None,
    ):  # TODO arguments and return type
        pyro.module("time_encoder", self.encoder)
        scale = u_scale / s_scale
        if self.latent_factor == "linear":
            encoder_weights = decoder_weights.T
            cell_code_loc = (
                torch.cat((u_obs - u_pcs_mean, s_obs - s_pcs_mean), dim=-1)
                @ encoder_weights
            )
            cell_code_loc = cell_code_loc.unsqueeze(-1).transpose(-1, -2)
            cell_code = pyro.sample(
                "cell_code",
                Normal(cell_code_loc, self.cell_code_scale).to_event(1),
            )
        with gene_plate:
            if self.latent_factor == "linear":
                regressor_output = torch.einsum(
                    "abc,cd->ad", cell_code, decoder_weights.squeeze()
                )
                regressor_u = softplus(
                    regressor_output[..., : self.num_genes].squeeze()
                    + u_pcs_mean
                )
                regressor_s = softplus(
                    regressor_output[..., self.num_genes :].squeeze()
                    + s_pcs_mean
                )
            if self.latent_factor_operation == "sum":
                u_obs = u_obs - regressor_u
                s_obs = s_obs - regressor_s
            if self.likelihood != "Normal":
                # u_ = torch.log1p(u_obs) / scale
                # s_ = torch.log1p(s_obs)
                u_ = u_obs / scale
                s_ = s_obs
            else:
                u_ = u_obs / scale
                s_ = s_obs

            tau = tau_inv(u_, s_, u0, s0, alpha, beta, gamma)
            # tau = torch.where(tau > dt_switching, dt_switching, tau)
            ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
            tau_ = tau_inv(u_, s_, u_inf, s_inf, self.zero, beta, gamma)
            ##tau_ = torch.where(tau_ >= tau_[s_obs > 0].max(dim=0)[0], tau_[s_obs > 0].max(dim=0)[0], tau_)
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
            switching = dt_switching + gene_offset
            t = torch.where(state, tau + gene_offset, tau_ + switching)
            ut = torch.where(state, ut, ut_)
            st = torch.where(state, st, st_)
            # cell_gene_state_logits = state_on-state_off
            # state = pyro.sample("cell_gene_state", Bernoulli(logits=cell_gene_state_logits)) == self.zero
            # gene_selection_logits = (ut - u_)**2 + (st - s_)**2
            # gene_selection = pyro.sample("gene_selection", Bernoulli(logits=gene_selection_logits)) == self.zero
            ##gene_selection_logits = torch.argsort(gene_selection_logits.sum(-2))
            ##gene_selection = alpha.new_zeros(alpha.shape)
            ##gene_selection[gene_selection_logits[:300]] = u0 + 1
            # t = t * gene_selection
        if self.shared_time:
            if not only_cell_times:
                cell_time_loc = torch.mean(
                    ((t - gene_offset) / t_scale)
                    if self.t_scale_on
                    else t,  # (t - gene_offset),
                    dim=gene_plate.dim,
                    keepdim=True,
                )
                cell_time_dist = Delta(cell_time_loc, event_dim=0)
                cell_time = pyro.sample("cell_time", cell_time_dist)
            else:
                # cell_time_loc, cell_time_scale = self.encoder(t)
                cell_time_loc = self.encoder(t)
                # cell_time_loc, u_read_depth_loc, u_read_depth_scale, s_read_depth_loc, s_read_depth_scale = self.encoder(t, u_log_library=u_log_library, s_log_library=s_log_library)
                # u_read_depth = pyro.sample('u_read_depth', LogNormal(u_read_depth_loc, u_read_depth_scale))
                # s_read_depth = pyro.sample('s_read_depth', LogNormal(s_read_depth_loc, s_read_depth_scale))
                cell_time = pyro.sample(
                    "cell_time", Delta(cell_time_loc, event_dim=0)
                )

                # cell_time_dist = TransformedDistribution(
                #            Normal(cell_time_loc, torch.sqrt(cell_time_scale)),
                #            #ExpTransform())  # equivalent to LogNormal(cell_time_loc, cell_time_scale), spelled out to clarify
                #            SoftplusTransform())
                # cell_time = pyro.sample("cell_time", cell_time_dist)

        # if self.likelihood != 'Normal':
        #    pyro.sample("ut", Normal(ut * scale, u_scale))
        #    pyro.sample("st", Normal(st, s_scale))

        with gene_plate:
            if not only_cell_times:
                if self.shared_time:
                    if self.t_scale_on:
                        t = t - cell_time * t_scale - gene_offset
                    else:
                        t = t - cell_time  # - gene_offset # FIX!
                pyro.sample("latent_time", Delta(t))

            # model:
            # cell_time[c] ~ ...
            # switching[g] ~ ...
            # t_noise[c, g] ~ ...
            # t[c, g] = cell_time[c] - switching[g] + t_noise[c, g]
            # u_obs | t[c, g] ~ ...
            #
            # guide (currently):
            # cell_time[c] ~ ...
            # switching[g] ~ ...
            # t[c, g] = f(u_obs)  # constant wrt cell_time
            # t_noise[c, g] = t[c, g] - cell_time[c] + switching[g]
            #
            # thus when replaying the guide through the model:
            # t[c, g] = cell_time[c] - switching[g] + (f(u_obs) - cell_time[c] + switching)
            #         = f(u_obs)  # constant wrt cell_time

            # updating the guide
            # t[c, g] = f(u_obs)
            # cell_time[c] = ((t[c, g] + switching[g]) / t_scale[g]).mean(g)
            # t_noise[c, g] = t[c, g] - cell_time[c] * t_scale[g]

            # cell-gene selection
            if self.latent_factor_operation == "selection":
                scale = u_scale / s_scale
                std_u = u_scale / scale
                velocity_error = ((ut - u_) / std_u) ** 2 + (
                    (st - s_) / s_scale
                ) ** 2
                pca_error = ((regressor_u / scale - u_) / std_u) ** 2 + (
                    (regressor_s - s_) / s_scale
                ) ** 2
                probs = torch.stack(
                    [
                        torch.exp(-velocity_error) - p_velocity,
                        torch.exp(-pca_error) - (1 - p_velocity),
                    ],
                    dim=-1,
                )
                probs = softmax(probs, dim=-1)  # probs.sum(-1, keepdim=True)
                p_gene_type = probs[..., 0]
                cellgene_type = pyro.sample(
                    "cellgene_type", Bernoulli(p_gene_type)
                )  ## tend to be constant
            if (
                self.latent_factor_operation == "selection"
                and self.latent_factor == "linear"
            ):
                return ut, st, u_, s_, regressor_u, regressor_s
            return ut, st, u_, s_, None, None


class VelocityAutoGuideList(AutoGuideList):
    def forward(self, *args, **kwargs):
        """
        A composite guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        if self.model.guide_type in ["velocity_auto", "velocity_auto_depth"]:
            pyro.module("time_encoder", self.model.time_encoder)
        if self.model.correct_library_size:  # use autoguide for read depth
            if self.model.guide_type != "velocity_auto_depth":
                pyro.module("u_lib_encoder", self.model.u_lib_encoder)
                pyro.module("s_lib_encoder", self.model.s_lib_encoder)
        result = super().forward(*args, **kwargs)
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result

        alpha = result["alpha"]
        beta = result["beta"]
        gamma = result["gamma"]

        if self.model.cell_specific_kinetics is not None:
            cell_state = args[-1]
            rho, _ = self.model.multikinetics_encoder(cell_state)
            alpha = rho * alpha

        if self.model.add_offset:
            u0 = result["u_offset"]
            s0 = result["s_offset"]
        else:
            u0 = s0 = alpha.new_zeros(alpha.shape)

        t0 = result["t0"]

        if "u_scale" in result:
            u_scale = result["u_scale"]
            s_scale = result["s_scale"]
        else:
            u_scale = s_scale = alpha.new_ones(alpha.shape)

        if self.model.cell_specific_kinetics is None:
            dt_switching = result["dt_switching"]
            u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
        else:
            dt_switching = None
            u_inf, s_inf = None, None

        with self.plates["cells"]:
            u_obs = args[0]
            s_obs = args[1]
            u_log_library = args[2]
            s_log_library = args[3]
            if self.model.correct_library_size:
                # if (self.model.guide_type != 'velocity_auto_depth'):
                #    u_lib_loc, u_lib_var = self.model.u_lib_encoder(torch.hstack([torch.log1p(u_obs), u_log_library]))
                #    s_lib_loc, s_lib_var = self.model.s_lib_encoder(torch.hstack([torch.log1p(s_obs), s_log_library]))
                #    result['u_read_depth'] = pyro.sample("u_read_depth", LogNormal(u_lib_loc, u_lib_var).to_event(0))
                #    result['s_read_depth'] = pyro.sample("s_read_depth", LogNormal(s_lib_loc, s_lib_var).to_event(0))
                if self.model.guide_type == "velocity_auto":
                    u_read_depth = None
                    s_read_depth = None
                if self.model.guide_type == "velocity_auto_depth":
                    u_read_depth = result["u_read_depth"]
                    s_read_depth = result["s_read_depth"]
            else:
                u_read_depth = s_read_depth = None
            t = self.model.get_time(
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
                u_read_depth=u_read_depth,
                s_read_depth=s_read_depth,
            )
            result["cell_time"] = t
        return result


class TrajectoryGuide(VelocityGuide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def guide(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        cell_plate = self.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        aux_cell_plate = self.plate("aux_cells", self.num_aux_cells, dim=-2)
        gene_plate = self.plate("genes", self.num_genes, dim=-1)

        with gene_plate, poutine.mask(mask=True):
            alpha = self.map_estimate("alpha")
            beta = self.map_estimate("beta")
            gamma = self.map_estimate("gamma")
            switching = self.map_estimate("switching")
            u_scale = self.map_estimate("u_scale")
            s_scale = self.map_estimate("s_scale")

        dt_aux = []
        u_aux = []
        s_aux = []
        zero = self.zero.to(u_obs.device)
        # u0 = s0 = zero
        u_prev = s_prev = zero

        for step in pyro.markov(range(aux_cell_plate.size)):
            dt_aux.append(self.map_estimate(f"dt_aux_{step}"))
            with gene_plate:
                u_noise = self.map_estimate(f"u_aux_noise_{step}")
                s_noise = self.map_estimate(f"s_aux_noise_{step}")
                state_aux = (
                    pyro.sample(
                        f"cell_gene_state_{step}",
                        Bernoulli(logits=dt_aux[step] - switching),
                    )
                    == zero
                )
                alpha = torch.where(state_aux, alpha, zero)
                u_loc, s_loc = mRNA(
                    dt_aux[step], u_prev, s_prev, alpha, beta, gamma
                )
                u_prev = u_loc + u_noise * torch.sqrt(2 * dt_aux[step])
                s_prev = s_loc + s_noise * torch.sqrt(2 * dt_aux[step])
                u_aux.append(u_prev)
                s_aux.append(s_prev)

        dt_aux = torch.hstack(dt_aux).to(u_obs.device)
        u_aux = torch.vstack(u_aux).to(u_obs.device)
        s_aux = torch.vstack(s_aux).to(u_obs.device)
        with cell_plate:
            # order_stat = torch.zeros(u_obs.shape[0], 1, u_aux.shape[0]).to(u_obs.device)
            # for i in range(order_stat.shape[0]):
            #    for j in range(order_stat.shape[1]):
            #        order_stat[i,0,j] = ((u_obs[i] - u_aux[j])**2+(s_obs[i] - s_aux[j])**2).sum()
            order_stat = (
                (u_obs.reshape(u_obs.shape[0], 1, u_obs.shape[1]) - u_aux) ** 2
            ).sum() + (
                (s_obs.reshape(s_obs.shape[0], 1, s_obs.shape[1]) - s_aux) ** 2
            ).sum(axis=-1)
            order_stat = order_stat.reshape(u_obs.shape[0], 1, u_aux.shape[0])
            order = pyro.sample(
                "order",
                Categorical(logits=-order_stat),
                infer={"enumerate": "parallel"},
            )
            dt = dt_aux[..., order]
            with gene_plate:
                u_noise = self.map_estimate("u_noise")
                s_noise = self.map_estimate("s_noise")
                u_prev = u_aux[..., order, :].squeeze()
                s_prev = s_aux[..., order, :].squeeze()
                u_loc_on, s_loc_on = mRNA(
                    dt, u_prev, s_prev, alpha, beta, gamma
                )
                u_loc_off, s_loc_off = mRNA(
                    dt, u_prev, s_prev, zero, beta, gamma
                )
                u_on = u_loc_on + u_noise * torch.sqrt(2 * dt)
                s_on = s_loc_on + s_noise * torch.sqrt(2 * dt)
                u_off = u_loc_off + u_noise * torch.sqrt(2 * dt)
                s_off = s_loc_off + s_noise * torch.sqrt(2 * dt)
                scale = u_scale / s_scale
                std_u = u_scale / scale
                mse_on = ((u_on - u_obs / scale) / std_u) ** 2 + (
                    (s_on - s_obs) / s_scale
                ) ** 2
                mse_off = ((u_off - u_obs / scale) / std_u) ** 2 + (
                    (s_off - s_obs) / s_scale
                ) ** 2
                state = pyro.sample(
                    "cell_gene_state", Bernoulli(logits=mse_on - mse_off)
                )


class DecoderTimeGuide(VelocityGuide):
    def __init__(self, model: VelocityModel):
        super().__init__(model)
        self.encoder = Encoder(self.num_genes * 2, n_output=10, n_layers=1)

    def guide(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        ind_x: Optional[torch.Tensor] = None,
    ):
        pyro.module("time_encoder", self.encoder)
        gene_plate = self.plate("genes", self.num_genes, dim=-1)
        cell_plate = self.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        u_ = torch.log1p(u_obs)
        s_ = torch.log1p(s_obs)
        x = torch.hstack([u_, s_])

        with gene_plate:
            switching = self.map_estimate("switching")
            alpha = self.map_estimate("alpha")
            beta = self.map_estimate("beta")
            gamma = self.map_estimate("gamma")

        with cell_plate:
            z_loc, z_scale, _ = self.encoder(x)
            z_loc = z_loc.unsqueeze(-2)
            z_scale = z_scale.unsqueeze(-2)
            pyro.sample(
                "latent_time_latent_space", Normal(z_loc, z_scale).to_event(1)
            )  # (cell, 1, components)


class KineticsParamEncoder(nn.Module):
    def __init__(self, hidden_shape, cell_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_shape),  # 2 for (u, s)
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_shape, 4),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(nn.Linear(cell_size, 1), nn.Softplus())
        # self.u_inf = nn.Sequential(
        #    nn.Linear(cell_size, 1),
        #    nn.Softplus()
        # )
        # self.s_inf = nn.Sequential(
        #    nn.Linear(cell_size, 1),
        #    nn.Softplus()
        # )

    def forward(self, u_obs, s_obs):
        kinetics_param = self.network(torch.stack([u_obs, s_obs], dim=-1))
        alpha = kinetics_param[..., 0].T
        beta = kinetics_param[..., 1].T
        gamma = kinetics_param[..., 2].T
        t = kinetics_param[..., 3].T

        alpha = self.linear(alpha).squeeze(-1)
        beta = self.linear(beta).squeeze(-1)
        gamma = self.linear(gamma).squeeze(-1)
        switching = self.linear(t).squeeze(-1)

        # u_inf = self.u_inf(u_obs.T).squeeze(-1)
        # s_inf = self.s_inf(s_obs.T).squeeze(-1)
        # return alpha, beta, gamma, u_inf, s_inf
        return alpha, beta, gamma, switching, softplus(t.T)


class GeneTimeEncoder(nn.Module):
    def __init__(self, hidden_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, hidden_shape),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_shape, 1),
        )

    def forward(
        self,
        u_obs,
        s_obs,
        alpha,
        alpha_off,
        beta,
        gamma,
        u0,
        s0,
        u_inf=None,
        s_inf=None,
    ):
        switching = self.network(
            torch.stack(
                [
                    u_inf,
                    s_inf,
                    u0.expand_as(u_inf),
                    s0.expand_as(s_inf),
                    alpha,
                    beta,
                    gamma,
                ],
                dim=-1,
            )
        )
        switching = switching[..., 0]
        tau = self.network(
            torch.stack(
                [
                    u_obs,
                    s_obs,
                    u0.expand_as(u_obs),
                    s0.expand_as(s_obs),
                    alpha.expand_as(u_obs),
                    beta.expand_as(s_obs),
                    gamma.expand_as(s_obs),
                ],
                dim=-1,
            )
        )
        tau = tau[..., 0]

        tau_ = self.network(
            torch.stack(
                [
                    u_obs,
                    s_obs,
                    u_inf.expand_as(u_obs),
                    s_inf.expand_as(s_obs),
                    torch.tensor(alpha_off).expand_as(u_obs),
                    beta.expand_as(u_obs),
                    gamma.expand_as(u_obs),
                ],
                dim=-1,
            )
        )
        tau_ = tau_[..., 0]
        # assert switching.shape[-1] == alpha.shape[-1]
        # assert tau.shape == u_obs.shape == s_obs.shape == tau_.shape
        return softplus(switching), softplus(tau), softplus(tau_)


class AutoDeltaRNAVelocityGuide(autoguide.AutoDelta):
    def __init__(self, model: VelocityModel, use_gpu: int, **initial_values):
        if initial_values != {}:
            init_loc_fn = autoguide.init_to_value(
                values={
                    key: value.to(f"cuda:{use_gpu}")
                    if key != "cell_time"
                    else pyro.subsample(
                        value.to(f"cuda:{use_gpu}"), event_dim=0
                    )
                    for key, value in initial_values.items()
                }
            )
            # return autoguide.init_to_value(site, values={site["name"]: pyro.subsample(self.cell_time_init, event_dim=self.event_dim)})
        else:
            init_loc_fn = autoguide.init_to_median
        super().__init__(
            poutine.block(model, hide_fn=site_is_discrete),
            init_loc_fn=init_loc_fn,
            create_plates=model.create_plates,
        )


class AutoNormalRNAVelocityGuide(autoguide.AutoNormal):
    def __init__(self, model: VelocityModel, use_gpu: int, **initial_values):
        if initial_values != {}:
            init_loc_fn = autoguide.init_to_value(
                values={
                    key: value.to(f"cuda:{use_gpu}")
                    for key, value in initial_values.items()
                }
            )
        else:
            init_loc_fn = autoguide.init_to_median
        super().__init__(
            poutine.block(model, hide_fn=site_is_discrete),
            init_loc_fn=init_loc_fn,
            create_plates=model.create_plates,
        )


# class LatentGuide(easyguide.EasyGuide):
#     def __init__(
#         self,
#         model: LatentFactor,
#         latent_factor_size: int = 10,
#         plate_size: int = 2,
#         inducing_point_size: int = 20,
#         **initial_values,
#     ):
#         super().__init__(model)
#         self.num_genes = model.num_genes
#         self.num_cells = model.num_cells
#         self.zero = model.zero
#         self.mask = initial_values.get("mask", None)
#         self.latent_factor_size = latent_factor_size
#         self.plate_size = plate_size
#         self.inducing_point_size = inducing_point_size

#         for key in initial_values:
#             self.register_buffer(f"{key}_init", initial_values[key])

#     def init(self, site):
#         if site["name"] == "cell_code":
#             return autoguide.init_to_value(
#                 site,
#                 values={site["name"]: pyro.subsample(self.cell_code_init, event_dim=1)},
#             )
#         if hasattr(self, f"{site['name']}_init"):
#             return autoguide.init_to_value(
#                 site, values={site["name"]: getattr(self, f"{site['name']}_init")}
#             )
#         return super().init(site)

#     def guide(
#         self,
#         u_obs: Optional[torch.Tensor] = None,
#         s_obs: Optional[torch.Tensor] = None,
#         u_log_library: Optional[torch.Tensor] = None,
#         s_log_library: Optional[torch.Tensor] = None,
#         ind_x: Optional[torch.Tensor] = None,
#     ):
#         """max plate = 2 cell plate and gene plate"""
#         if self.plate_size == 2:
#             decoder_weights = self.map_estimate("cell_codebook")
#             encoder_weights = decoder_weights.T
#             gene_plate = self.plate("genes", self.num_genes, dim=-1)
#             with gene_plate:
#                 u_scale = self.map_estimate("u_scale")
#                 s_scale = self.map_estimate("s_scale")
#                 u_pcs_mean = self.map_estimate("u_pcs_mean")
#                 s_pcs_mean = self.map_estimate("s_pcs_mean")

#             if self.inducing_point_size == 0:
#                 cell_code_scale = pyro.param(
#                     "cell_code_scale", lambda: torch.tensor(0.1), constraint=positive
#                 ).to(u_obs.device)
#             cell_plate = self.plate("cells", self.num_cells, dim=-2, subsample=ind_x)
#             with cell_plate:
#                 cell_code_loc = (
#                     torch.cat((u_obs - u_pcs_mean, s_obs - s_pcs_mean), dim=-1)
#                     @ encoder_weights
#                 )
#                 if self.inducing_point_size > 0:
#                     inducing_points = pyro.param(
#                         "inducing_points",
#                         lambda: torch.randn(
#                             self.inducing_point_size, self.latent_factor_size
#                         ),
#                     ).to(u_obs.device)
#                     inducing_mean = pyro.param(
#                         "inducing_mean",
#                         lambda: torch.randn(
#                             self.inducing_point_size, self.latent_factor_size
#                         ),
#                     ).to(u_obs.device)

#                     kernel = pyro.contrib.gp.kernels.RBF(
#                         input_dim=self.latent_factor_size
#                     )
#                     assert cell_code_loc.shape == (
#                         cell_plate.subsample_size,
#                         self.latent_factor_size,
#                     )
#                     assert inducing_points.shape == (
#                         self.inducing_point_size,
#                         self.latent_factor_size,
#                     )
#                     assert inducing_mean.shape == (
#                         self.inducing_point_size,
#                         self.latent_factor_size,
#                     )
#                     cell_code_loc, cell_code_scale = pyro.contrib.gp.util.conditional(
#                         cell_code_loc,
#                         inducing_points,
#                         kernel,
#                         inducing_mean.transpose(-1, -2),
#                         full_cov=False,
#                     )
#                     assert cell_code_loc.shape == (
#                         self.latent_factor_size,
#                         cell_plate.subsample_size,
#                     )
#                     cell_code_loc = cell_code_loc.T.unsqueeze(-1).transpose(-1, -2)
#                     cell_code_scale = cell_code_scale.T.unsqueeze(-1).transpose(-1, -2)
#                 else:
#                     cell_code_loc = cell_code_loc.unsqueeze(-1).transpose(-1, -2)
#                 cell_code = pyro.sample(
#                     "cell_code", Normal(cell_code_loc, cell_code_scale).to_event(1)
#                 )
#         else:
#             cell_code = self.guide2(u_obs, s_obs, u_log_library, s_log_library, ind_x)
#         return cell_code

#     def guide2(
#         self,
#         u_obs: Optional[torch.Tensor] = None,
#         s_obs: Optional[torch.Tensor] = None,
#         u_log_library: Optional[torch.Tensor] = None,
#         s_log_library: Optional[torch.Tensor] = None,
#         ind_x: Optional[torch.Tensor] = None,
#     ):
#         """max plate = 1 only cell plate"""
#         cell_plate = self.plate("cells", self.num_cells, subsample=ind_x, dim=-1)
#         decoder_weights = self.map_estimate("cell_codebook")
#         encoder_weights = decoder_weights.T
#         u_scale = self.map_estimate("u_scale")
#         s_scale = self.map_estimate("s_scale")
#         u_pcs_mean = self.map_estimate("u_pcs_mean")
#         s_pcs_mean = self.map_estimate("s_pcs_mean")
#         cell_plate = self.plate("cells", self.num_cells, subsample=ind_x, dim=-1)
#         # cell_code_scale = pyro.param("cell_code_scale", lambda: torch.ones(self.latent_factor_size) * 0.1, constraint=positive, event_dim=1).to(u_obs.device)
#         cell_code_scale = pyro.param(
#             "cell_code_scale", lambda: torch.tensor(0.1), constraint=positive
#         ).to(u_obs.device)
#         with cell_plate:
#             cell_code = (
#                 torch.cat((u_obs - u_pcs_mean, s_obs - s_pcs_mean), dim=-1)
#                 @ encoder_weights
#             )
#             cell_code = pyro.sample(
#                 "cell_code", Normal(cell_code, cell_code_scale).to_event(1)
#             )
#         return cell_code
