from typing import Optional, Tuple, Union

import pyro
import torch
from packaging import version
from pyro import poutine
from pyro.distributions import (
    Bernoulli,
    Beta,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
)
from pyro.distributions.constraints import positive
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.primitives import plate
from scvi.nn import Decoder
from torch.nn.functional import relu, softplus

from .utils import mRNA


class LogNormalModel(PyroModule):
    """
    A base class for pyrovelocity models.

    This class serves as the base class for constructing a pyrovelocity model.
    It provides basic methods for handling gene expression data, such as creating plates
    for cells and genes, encoding cell-specific features, and computing the likelihood
    of the observed data.

    Attributes:
        num_cells (int): The number of cells.
        num_genes (int): The number of genes.
        likelihood (str): The likelihood type for the model, defaults to "Poisson".
        plate_size (int): The size of the plate for the model, defaults to 2.

    Example:
        >>> from pyrovelocity._velocity_model import LogNormalModel
        >>> num_cells = 10
        >>> num_genes = 20
        >>> likelihood = "Poisson"
        >>> plate_size = 2
        >>> model = LogNormalModel(num_cells, num_genes, likelihood, plate_size)
        >>> assert model.num_cells == num_cells
        >>> assert model.num_genes == num_genes
        >>> assert model.likelihood == likelihood
        >>> assert model.plate_size == plate_size
    """

    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        likelihood: str = "Poisson",
        plate_size: int = 2,
        correct_library_size: Union[bool, str] = True,
    ) -> None:
        assert num_cells > 0 and num_genes > 0
        super().__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.n_obs = None
        self.plate_size = plate_size
        self.correct_library_size = correct_library_size
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
            return LogNormal(self.zero, self.one).mask(self.include_prior)
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
        """
        Compute the likelihood of the given count data.

        Args:
            ut (torch.Tensor): Tensor representing unspliced transcripts.
            st (torch.Tensor): Tensor representing spliced transcripts.
            u_log_library (Optional[torch.Tensor], optional): Log library tensor for unspliced transcripts. Defaults to None.
            s_log_library (Optional[torch.Tensor], optional): Log library tensor for spliced transcripts. Defaults to None.
            u_scale (Optional[torch.Tensor], optional): Scale tensor for unspliced transcripts. Defaults to None.
            s_scale (Optional[torch.Tensor], optional): Scale tensor for spliced transcripts. Defaults to None.
            u_read_depth (Optional[torch.Tensor], optional): Read depth tensor for unspliced transcripts. Defaults to None.
            s_read_depth (Optional[torch.Tensor], optional): Read depth tensor for spliced transcripts. Defaults to None.
            u_cell_size_coef (Optional[Any], optional): Cell size coefficient for unspliced transcripts. Defaults to None.
            ut_coef (Optional[Any], optional): Coefficient for unspliced transcripts. Defaults to None.
            s_cell_size_coef (Optional[Any], optional): Cell size coefficient for spliced transcripts. Defaults to None.
            st_coef (Optional[Any], optional): Coefficient for spliced transcripts. Defaults to None.

        Returns:
            Tuple[Poisson, Poisson]: A tuple of Poisson distributions for unspliced and spliced transcripts, respectively.

        Example:
            >>> import torch
            >>> from pyro.nn import PyroModule
            >>> num_cells = 10
            >>> num_genes = 20
            >>> likelihood = "Poisson"
            >>> plate_size = 2
            >>> model = LogNormalModel(num_cells, num_genes, likelihood, plate_size)
            >>> ut = torch.rand(num_cells, num_genes)
            >>> st = torch.rand(num_cells, num_genes)
            >>> u_read_depth = torch.rand(num_cells, 1)
            >>> s_read_depth = torch.rand(num_cells, 1)
            >>> u_dist, s_dist = model.get_likelihood(ut, st, u_read_depth=u_read_depth, s_read_depth=s_read_depth)
            >>> assert isinstance(u_dist, torch.distributions.Poisson)
            >>> assert isinstance(s_dist, torch.distributions.Poisson)
        """
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
                if self.correct_library_size == "cell_size_regress":
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
                u_dist = Normal(ut, u_scale)
                s_dist = Normal(st, s_scale)
            else:
                u_dist = Normal(ut, self.one * 0.1)
                s_dist = Normal(st, self.one * 0.1)
        elif self.likelihood == "LogNormal":
            if u_scale is not None and s_scale is not None:
                u_dist = LogNormal((ut + self.one * 1e-6).log(), u_scale)
                s_dist = LogNormal((st + self.one * 1e-6).log(), s_scale)
            else:
                u_dist = LogNormal(ut, self.one * 0.1)
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
        """
        Derived class for modeling RNA velocity with shared latent time.

        Inherits from LogNormalModel.

        Args:
            num_cells (int): Number of cells.
            num_genes (int): Number of genes.
            likelihood (str, optional): Likelihood model. Defaults to "Poisson".
            shared_time (bool, optional): Whether to share time across cells. Defaults to True.
            t_scale_on (bool, optional): Whether to scale the time. Defaults to False.
            plate_size (int, optional): The number of dimensions for the plates. Defaults to 2.
            latent_factor (str, optional): Latent factor type. Defaults to "none".
            latent_factor_size (int, optional): Latent factor size. Defaults to 30.
            latent_factor_operation (str, optional): Latent factor operation. Defaults to "selection".
            include_prior (bool, optional): Whether to include prior. Defaults to False.
            num_aux_cells (int, optional): Number of auxiliary cells. Defaults to 100.
            only_cell_times (bool, optional): Whether to use only cell times. Defaults to False.
            decoder_on (bool, optional): Whether to use a decoder. Defaults to False.
            add_offset (bool, optional): Whether to add an offset. Defaults to False.
            correct_library_size (Union[bool, str], optional): Whether to correct library size. Defaults to True.
            guide_type (bool, optional): Guide type for the model. Defaults to "velocity".
            cell_specific_kinetics (Optional[str], optional): Cell-specific kinetics. Defaults to None.
            kinetics_num (Optional[int], optional): Number of kinetics. Defaults to None.
            **initial_values: Initial values for various parameters in the model.

        Example:
            >>> import torch
            >>> from pyro.nn import PyroModule
            >>> num_cells = 10
            >>> num_genes = 20
            >>> likelihood = "Poisson"
            >>> plate_size = 2
            >>> model = VelocityModel(num_cells, num_genes, likelihood, plate_size)
            >>> u_obs = torch.rand(num_cells, num_genes)
            >>> s_obs = torch.rand(num_cells, num_genes)
            >>> u_log_library = torch.rand(num_cells, 1)
            >>> s_log_library = torch.rand(num_cells, 1)
            >>> ind_x = torch.randint(0, num_cells, (num_cells,))
            >>> # ut, st = model.model(u_obs, s_obs, u_log_library, s_log_library, ind_x)
            >>> # assert ut.shape == (num_cells, num_genes)
            >>> # assert st.shape == (num_cells, num_genes)
        """
        assert num_cells > 0 and num_genes > 0
        super().__init__(num_cells, num_genes, likelihood, plate_size)
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


class AuxCellVelocityModel(VelocityModel):
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
            cell_time = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )

        with gene_plate:
            if self.latent_factor_operation == "selection":
                cellgene_type = pyro.sample(
                    "cellgene_type", Bernoulli(p_velocity)
                )
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

            ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)

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
        return ut, st


class VelocityModelAuto(AuxCellVelocityModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
        """
        Computes the unspliced (u) and spliced (s) RNA expression levels given the model parameters.

        Args:
            u_scale (torch.Tensor): Scaling factor for unspliced expression.
            s_scale (torch.Tensor): Scaling factor for spliced expression.
            alpha (torch.Tensor): Transcription rate.
            beta (torch.Tensor): Splicing rate.
            gamma (torch.Tensor): Degradation rate.
            t (torch.Tensor): Cell time.
            u0 (torch.Tensor): Unspliced RNA initial expression.
            s0 (torch.Tensor): Spliced RNA initial expression.
            t0 (torch.Tensor): Initial cell time.
            switching (Optional[torch.Tensor], optional): Switching time. Default is None.
            u_inf (Optional[torch.Tensor], optional): Unspliced RNA expression at switching time. Default is None.
            s_inf (Optional[torch.Tensor], optional): Spliced RNA expression at switching time. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The unspliced (u) and spliced (s) RNA expression levels.

        Examples:
            >>> import torch
            >>> from pyrovelocity._velocity_model import VelocityModelAuto
            >>> model = VelocityModelAuto(
            ...             3,
            ...             4,
            ...             "Poisson",
            ...             True,
            ...             False,
            ...             2,
            ...             "none",
            ...             latent_factor_operation="selection",
            ...             latent_factor_size=10,
            ...             include_prior=False,
            ...             num_aux_cells=0,
            ...             only_cell_times=True,
            ...             decoder_on=False,
            ...             add_offset=False,
            ...             correct_library_size=True,
            ...             guide_type="auto_t0_constraint",
            ...             cell_specific_kinetics=None,
            ...             **{}
            ...         )
            >>> u, s = model.get_rna(
            ...            u_scale=torch.tensor([0.9793, 1.0567, 0.8610, 0.9304], device="cpu"),
            ...            s_scale=torch.tensor(1.0),
            ...            alpha=torch.tensor([0.4869, 1.5997, 1.3962, 0.5038], device="cpu"),
            ...            beta=torch.tensor([0.5403, 1.1192, 0.9912, 1.1783], device="cpu"),
            ...            gamma=torch.tensor([1.9612, 0.5533, 2.1050, 4.9345], device="cpu"),
            ...            t=torch.tensor([[0.4230], [0.5119], [0.2689]], device="cpu"),
            ...            u0=torch.tensor(0.0),
            ...            s0=torch.tensor(0.0),
            ...            t0=torch.tensor([-0.4867, 0.5581, -0.6957, 0.6028], device="cpu"),
            ...            switching=torch.tensor([1.1886, 1.1227, 0.6789, 4.1003], device="cpu"),
            ...            u_inf=torch.tensor([0.5367, 0.6695, 1.0479, 0.4206], device="cpu"),
            ...            s_inf=torch.tensor([0.1132, 0.2100, 0.3750, 0.0999], device="cpu"),
            >>>        )
        """
        if self.cell_specific_kinetics is None:
            if (
                self.guide_type != "auto"
                and version.parse(pyro.__version__) > version.parse("1.8.1")
                or self.guide_type == "auto"
            ):
                enum = "parallel"
            elif (
                self.guide_type != "auto"
                and not pyro.__version__.startswith("1.8.1")
                and pyro.__version__.startswith("1.6.0")
            ):
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
        cell_state: Optional[torch.Tensor] = None,
        time_info: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward model, which computes the unspliced (u) and spliced (s) RNA expression levels given the observations and model parameters.

        Args:
            u_obs (Optional[torch.Tensor], optional): Observed unspliced RNA expression. Default is None.
            s_obs (Optional[torch.Tensor], optional): Observed spliced RNA expression. Default is None.
            u_log_library (Optional[torch.Tensor], optional): Log-transformed library size for unspliced RNA. Default is None.
            s_log_library (Optional[torch.Tensor], optional): Log-transformed library size for spliced RNA. Default is None.
            u_log_library_loc (Optional[torch.Tensor], optional): Mean of log-transformed library size for unspliced RNA. Default is None.
            s_log_library_loc (Optional[torch.Tensor], optional): Mean of log-transformed library size for spliced RNA. Default is None.
            u_log_library_scale (Optional[torch.Tensor], optional): Scale of log-transformed library size for unspliced RNA. Default is None.
            s_log_library_scale (Optional[torch.Tensor], optional): Scale of log-transformed library size for spliced RNA. Default is None.
            ind_x (Optional[torch.Tensor], optional): Indices for the cells. Default is None.
            cell_state (Optional[torch.Tensor], optional): Cell state information. Default is None.
            time_info (Optional[torch.Tensor], optional): Time information for the cells. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The unspliced (u) and spliced (s) RNA expression levels.

        Examples:
            >>> import torch
            >>> from pyrovelocity._velocity_model import VelocityModelAuto
            >>> u_obs=torch.tensor(
            ...     [[33.,  1.,  7.,  1.],
            ...     [12., 30., 11.,  3.],
            ...     [ 1.,  1.,  8.,  5.]],
            ...     device="cpu",
            >>> )
            >>> s_obs=torch.tensor(
            ...     [[32.0, 0.0, 6.0, 0.0],
            ...     [11.0, 29.0, 10.0, 2.0],
            ...     [0.0, 0.0, 7.0, 4.0]],
            ...     device="cpu",
            >>> )
            >>> u_log_library=torch.tensor([[3.7377], [4.0254], [2.7081]], device="cpu")
            >>> s_log_library=torch.tensor([[3.6376], [3.9512], [2.3979]], device="cpu")
            >>> u_log_library_loc=torch.tensor([[3.4904], [3.4904], [3.4904]], device="cpu")
            >>> s_log_library_loc=torch.tensor([[3.3289], [3.3289], [3.3289]], device="cpu")
            >>> u_log_library_scale=torch.tensor([[0.6926], [0.6926], [0.6926]], device="cpu")
            >>> s_log_library_scale=torch.tensor([[0.8214], [0.8214], [0.8214]], device="cpu")
            >>> ind_x=torch.tensor([2, 0, 1], device="cpu")
            >>> model = VelocityModelAuto(3,4)
            >>> u, s = model.forward(
            >>>            u_obs,
            >>>            s_obs,
            >>>            u_log_library,
            >>>            s_log_library,
            >>>            u_log_library_loc,
            >>>            s_log_library_loc,
            >>>            u_log_library_scale,
            >>>            s_log_library_scale,
            >>>            ind_x,
            >>>        )
            >>> u, s
            (tensor([[33.,  1.,  7.,  1.],
                    [12., 30., 11.,  3.],
                    [ 1.,  1.,  8.,  5.]]),
            tensor([[32.,  0.,  6.,  0.],
                    [11., 29., 10.,  2.],
                    [ 0.,  0.,  7.,  4.]]))
        """

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
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )

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
                if self.guide_type in [
                    "auto_t0_constraint",
                    "velocity_auto_t0_constraint",
                ]:
                    pyro.sample(
                        "time_constraint",
                        Bernoulli(logits=t - t0),
                        obs=self.one,
                    )
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
