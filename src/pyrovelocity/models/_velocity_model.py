from typing import Optional
from typing import Tuple
from typing import Union

import pyro
import torch
from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
from pyro import poutine
from pyro.distributions import Bernoulli
from pyro.distributions import LogNormal
from pyro.distributions import Normal
from pyro.distributions import Poisson
from pyro.nn import PyroModule
from pyro.nn import PyroSample
from pyro.primitives import plate
from scvi.nn import Decoder
from torch.nn.functional import relu
from torch.nn.functional import softplus

from pyrovelocity.logging import configure_logging
from pyrovelocity.models._transcription_dynamics import mrna_dynamics


logger = configure_logging(__name__)

RNAInputType = Union[
    Float[torch.Tensor, ""],
    Float[torch.Tensor, "num_genes"],
    Float[torch.Tensor, "samples num_genes"],
]

RNAOutputType = Union[
    Float[torch.Tensor, "num_cells num_genes"],
    Float[torch.Tensor, "samples num_cells num_genes"],
]


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
        >>> from pyrovelocity.models._velocity_model import LogNormalModel
        >>> num_cells = 3
        >>> num_genes = 4
        >>> likelihood = "Poisson"
        >>> plate_size = 2
        >>> model = LogNormalModel(num_cells, num_genes, likelihood, plate_size)
        >>> logger.info(model)
        >>> assert model.num_cells == num_cells
        >>> assert model.num_genes == num_genes
        >>> assert model.likelihood == likelihood
        >>> assert model.plate_size == plate_size
    """

    @beartype
    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        likelihood: str = "Poisson",
        plate_size: int = 2,
        correct_library_size: Union[bool, str] = True,
    ) -> None:
        """
        Initialize the LogNormalModel.

        TODO: make the likelihood parameter a sum type over supported distributions.

        Args:
            num_cells (int): number of cells.
            num_genes (int): number of genes.
            likelihood (str, optional): Type of likelihood to use. Defaults to "Poisson".
            plate_size (int, optional): Number of plates. Defaults to 2.
            correct_library_size (Union[bool, str], optional): Flag to control normalization by library size. Defaults to True.
        """
        logger.info("Initializing LogNormalModel")
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

    @beartype
    def __repr__(self) -> str:
        return (
            f"\nLogNormalModel(\n"
            f"\tnum_cells={self.num_cells}, \n"
            f"\tnum_genes={self.num_genes}, \n"
            f"\tlikelihood={self.likelihood}, \n"
            f"\tplate_size={self.plate_size}, \n"
            f"\tcorrect_library_size={self.correct_library_size}\n"
            f")\n"
        )

    @beartype
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
        """
        Create cell and gene plates for the model. note that usage in
        pyro.infer.autoguide.AutoGuideList requires the set of arguments be
        equivalent to the model's forward method. See
        https://github.com/pyro-ppl/pyro/blob/670e9cb57003b52aea02deeb1316a42db3e1de1f/pyro/infer/autoguide/guides.py#L60-L63.

        Args:
            u_obs (Optional[torch.Tensor], optional): _description_. Defaults to None.
            s_obs (Optional[torch.Tensor], optional): _description_. Defaults to None.
            u_log_library (Optional[torch.Tensor], optional): _description_. Defaults to None.
            s_log_library (Optional[torch.Tensor], optional): _description_. Defaults to None.
            u_log_library_loc (Optional[torch.Tensor], optional): _description_. Defaults to None.
            s_log_library_loc (Optional[torch.Tensor], optional): _description_. Defaults to None.
            u_log_library_scale (Optional[torch.Tensor], optional): _description_. Defaults to None.
            s_log_library_scale (Optional[torch.Tensor], optional): _description_. Defaults to None.
            ind_x (Optional[torch.Tensor], optional): _description_. Defaults to None.
            cell_state (Optional[torch.Tensor], optional): _description_. Defaults to None.
            time_info (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            Tuple[plate, plate]: _description_
        """
        cell_plate = pyro.plate(
            "cells", self.num_cells, subsample=ind_x, dim=-2
        )
        gene_plate = pyro.plate("genes", self.num_genes, dim=-1)
        return cell_plate, gene_plate

    @PyroSample
    def alpha(self):
        return self._pyrosample_helper(1.0)

    @PyroSample
    def beta(self):
        return self._pyrosample_helper(0.25)

    @PyroSample
    def gamma(self):
        return self._pyrosample_helper(1.0)

    @PyroSample
    def u_scale(self):
        return self._pyrosample_helper(0.1)

    @PyroSample
    def s_scale(self):
        return self._pyrosample_helper(0.1)

    @PyroSample
    def u_inf(self):
        return self._pyrosample_helper(0.1)

    @PyroSample
    def s_inf(self):
        return self._pyrosample_helper(0.1)

    @PyroSample
    def dt_switching(self):
        return self._pyrosample_helper(1.0)

    @PyroSample
    def gene_offset(self):
        return Normal(self.zero, self.one)

    @PyroSample
    def t_scale(self):
        return Normal(self.zero, self.one * 0.1)

    @PyroSample
    def latent_time(self):
        if self.shared_time & self.plate_size == 2:
            return Normal(self.zero, self.one * 0.1).mask(self.include_prior)
        else:
            return (
                LogNormal(self.zero, self.one)
                .expand((self.num_genes,))
                .to_event(1)
            )

    @PyroSample
    def cell_time(self):
        if self.plate_size == 2 and self.shared_time:
            return Normal(self.zero, self.one)
        else:
            return LogNormal(self.zero, self.one)

    def _pyrosample_helper(self, scale: float):
        if self.plate_size == 2:
            return LogNormal(self.zero, self.one * scale)
        return (
            LogNormal(self.zero, self.one * 0.1)
            .expand((self.num_genes,))
            .to_event(1)
            .mask(False)
        )

    @beartype
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
            >>> from pyrovelocity.models._velocity_model import LogNormalModel
            >>> num_cells = 10
            >>> num_genes = 20
            >>> likelihood = "Poisson"
            >>> plate_size = 2
            >>> model = LogNormalModel(num_cells, num_genes, likelihood, plate_size)
            >>> logger.info(model)
            >>> ut = torch.rand(num_cells, num_genes)
            >>> st = torch.rand(num_cells, num_genes)
            >>> u_read_depth = torch.rand(num_cells, 1)
            >>> s_read_depth = torch.rand(num_cells, 1)
            >>> u_dist, s_dist = model.get_likelihood(ut, st, u_read_depth=u_read_depth, s_read_depth=s_read_depth)
            >>> logger.info(f"u_dist: {u_dist}")
            >>> logger.info(f"s_dist: {s_dist}")
            >>> assert isinstance(u_dist, torch.distributions.Poisson)
            >>> assert isinstance(s_dist, torch.distributions.Poisson)
        """
        if self.likelihood != "Poisson":
            likelihood_not_implemented_msg = (
                "In the future, the likelihood will be referred to via a "
                "member of a sum type over supported distributions"
            )
            raise NotImplementedError(likelihood_not_implemented_msg)

        if self.correct_library_size:
            ut = relu(ut) + self.one * 1e-6
            st = relu(st) + self.one * 1e-6
            ut = pyro.deterministic("ut", ut, event_dim=0)
            st = pyro.deterministic("st", st, event_dim=0)
            ut = ut / torch.sum(ut, dim=-1, keepdim=True)
            st = st / torch.sum(st, dim=-1, keepdim=True)
            ut = pyro.deterministic("ut_norm", ut, event_dim=0)
            st = pyro.deterministic("st_norm", st, event_dim=0)
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
        return u_dist, s_dist


class VelocityModelAuto(LogNormalModel):
    """Automatically configured velocity model.

    Args:
        num_cells (int): _description_
        num_genes (int): _description_
        likelihood (str, optional): _description_. Defaults to "Poisson".
        shared_time (bool, optional): _description_. Defaults to True.
        t_scale_on (bool, optional): _description_. Defaults to False.
        plate_size (int, optional): _description_. Defaults to 2.
        latent_factor (str, optional): _description_. Defaults to "none".
        latent_factor_size (int, optional): _description_. Defaults to 30.
        latent_factor_operation (str, optional): _description_. Defaults to "selection".
        include_prior (bool, optional): _description_. Defaults to False.
        num_aux_cells (int, optional): _description_. Defaults to 100.
        only_cell_times (bool, optional): _description_. Defaults to False.
        decoder_on (bool, optional): _description_. Defaults to False.
        add_offset (bool, optional): _description_. Defaults to False.
        correct_library_size (Union[bool, str], optional): _description_. Defaults to True.
        guide_type (str, optional): _description_. Defaults to "velocity".
        cell_specific_kinetics (Optional[str], optional): _description_. Defaults to None.
        kinetics_num (Optional[int], optional): _description_. Defaults to None.

    Examples:
        >>> import torch
        >>> from pyrovelocity.models._velocity_model import VelocityModelAuto
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
        >>> logger.info(model)
    """

    @beartype
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
        guide_type: str = "velocity",
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        **initial_values,
    ) -> None:
        assert num_cells > 0 and num_genes > 0
        super().__init__(num_cells, num_genes, likelihood, plate_size)
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

        self.enumeration = "parallel"
        # self.set_enumeration_strategy()

    def sample_cell_gene_state(self, t, switching):
        return (
            pyro.sample(
                "cell_gene_state",
                Bernoulli(logits=t - switching),
                infer={"enumerate": self.enumeration},
            )
            == self.zero
        )

    @beartype
    def __repr__(self) -> str:
        return (
            f"\nVelocityModelAuto(\n"
            f"\tnum_cells={self.num_cells}, \n"
            f"\tnum_genes={self.num_genes}, \n"
            f'\tlikelihood="{self.likelihood}", \n'
            f"\tshared_time={self.shared_time}, \n"
            f"\tt_scale_on={self.t_scale_on}, \n"
            f"\tplate_size={self.plate_size}, \n"
            f'\tlatent_factor="{self.latent_factor}", \n'
            f"\tlatent_factor_size={self.latent_factor_size}, \n"
            f'\tlatent_factor_operation="{self.latent_factor_operation}", \n'
            f"\tinclude_prior={self.include_prior}, \n"
            f"\tnum_aux_cells={self.num_aux_cells}, \n"
            f"\tonly_cell_times={self.only_cell_times}, \n"
            f"\tdecoder_on={self.decoder_on}, \n"
            f"\tadd_offset={self.add_offset}, \n"
            f"\tcorrect_library_size={self.correct_library_size}, \n"
            f'\tguide_type="{self.guide_type}", \n'
            f"\tcell_specific_kinetics={self.cell_specific_kinetics}, \n"
            f"\tkinetics_num={self.k}\n"
            f")\n"
        )

    @jaxtyped(typechecker=beartype)
    def get_rna(
        self,
        u_scale: RNAInputType,
        s_scale: RNAInputType,
        alpha: RNAInputType,
        beta: RNAInputType,
        gamma: RNAInputType,
        t: Float[torch.Tensor, "num_cells time"],
        u0: RNAInputType,
        s0: RNAInputType,
        t0: RNAInputType,
        switching: Optional[RNAInputType] = None,
        u_inf: Optional[RNAInputType] = None,
        s_inf: Optional[RNAInputType] = None,
    ) -> Tuple[RNAOutputType, RNAOutputType]:
        """
        Computes the unspliced (u) and spliced (s) RNA expression levels given
        the model parameters.

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
            >>> from pyrovelocity.models._velocity_model import VelocityModelAuto
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
            ...            s_scale=torch.tensor(1.),
            ...            alpha=torch.tensor([0.4869, 1.5997, 1.3962, 0.5038], device="cpu"),
            ...            beta=torch.tensor([0.5403, 1.1192, 0.9912, 1.1783], device="cpu"),
            ...            gamma=torch.tensor([1.9612, 0.5533, 2.1050, 4.9345], device="cpu"),
            ...            t=torch.tensor([[0.4230], [0.5119], [0.2689]], device="cpu"),
            ...            u0=torch.tensor(0.),
            ...            s0=torch.tensor(0.),
            ...            t0=torch.tensor([-0.4867, 0.5581, -0.6957, 0.6028], device="cpu"),
            ...            switching=torch.tensor([1.1886, 1.1227, 0.6789, 4.1003], device="cpu"),
            ...            u_inf=torch.tensor([0.5367, 0.6695, 1.0479, 0.4206], device="cpu"),
            ...            s_inf=torch.tensor([0.1132, 0.2100, 0.3750, 0.0999], device="cpu"),
            >>>        )
            >>> logger.info(f"u: {u}")
            >>> logger.info(f"s: {s}")
        """
        state = self.sample_cell_gene_state(t, switching)
        alpha_off = self.zero
        u0_vec = torch.where(state, u0, u_inf)
        s0_vec = torch.where(state, s0, s_inf)
        alpha_vec = torch.where(state, alpha, alpha_off)
        tau = softplus(torch.where(state, t - t0, t - switching))

        ut, st = mrna_dynamics(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
        ut = ut * u_scale / s_scale
        return ut, st

    @beartype
    def forward(
        self,
        u_obs: torch.Tensor,
        s_obs: torch.Tensor,
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
        Defines the forward model, which computes the unspliced (u) and spliced
        (s) RNA expression levels given the observations and model parameters.

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
            >>> from pyrovelocity.models._velocity_model import VelocityModelAuto
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

            u_scale = self.u_scale
            s_scale = self.one

            dt_switching = self.dt_switching
            u_inf, s_inf = mrna_dynamics(
                dt_switching, u0, s0, alpha, beta, gamma
            )
            u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
            s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
            switching = pyro.deterministic(
                "switching", dt_switching + t0, event_dim=0
            )

        with cell_plate:
            t = pyro.sample(
                "cell_time",
                LogNormal(self.zero, self.one).mask(self.include_prior),
            )

        with cell_plate:
            u_cell_size_coef = ut_coef = s_cell_size_coef = st_coef = None
            u_read_depth = pyro.sample(
                "u_read_depth", LogNormal(u_log_library, u_log_library_scale)
            )
            s_read_depth = pyro.sample(
                "s_read_depth", LogNormal(s_log_library, s_log_library_scale)
            )
            with gene_plate:
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
