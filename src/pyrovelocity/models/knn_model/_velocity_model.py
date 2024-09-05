from typing import Optional, Tuple, Union

import pyro
from pyro.nn import PyroModule
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from pyro import poutine
from pyro.distributions import Bernoulli, LogNormal, Normal, Poisson
from pyro.nn import PyroModule, PyroSample
from pyro.primitives import plate
from scvi.nn import Decoder
from torch.nn.functional import relu, softplus
from torch import Tensor

from pyrovelocity.logging import configure_logging
from pyrovelocity.models._transcription_dynamics import mrna_dynamics, atac_mrna_dynamics, get_initial_states, get_cell_parameters

from pyrovelocity.models.knn_model.regulatory_functions_torch import regulatory_function_1
from pyrovelocity.models.knn_model._vector_fields import vector_field_1

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

__all__ = [
    "VelocityModelAuto",
    "MultiVelocityModelAuto",
]

class VelocityModelAuto(PyroModule):
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
        n_batch: int,
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
        stochastic_v_ag_hyp_prior={"alpha": 6.0, "beta": 3.0},
        s_overdispersion_factor_hyp_prior={'alpha_mean': 100., 'beta_mean': 1.,
                                           'alpha_sd': 1., 'beta_sd': 0.1},
        detection_hyp_prior={"alpha": 10.0, "mean_alpha": 1.0, "mean_beta": 1.0},
        detection_i_prior={"mean": 1, "alpha": 100},
        detection_gi_prior={"mean": 1, "alpha": 200},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={"alpha": 1.0, "beta": 100.0},
        Tmax_prior={"mean": 50., "sd": 50.},
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
        
        self.n_obs = num_cells
        self.n_vars = num_genes
        self.n_batch = n_batch
        
        self.stochastic_v_ag_hyp_prior = stochastic_v_ag_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.detection_hyp_prior = detection_hyp_prior
        self.s_overdispersion_factor_hyp_prior = s_overdispersion_factor_hyp_prior
        self.detection_gi_prior = detection_gi_prior
        self.detection_i_prior = detection_i_prior
        
        self.register_buffer(
            "s_overdispersion_factor_alpha_mean",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["alpha_mean"]),
        )
        self.register_buffer(
            "s_overdispersion_factor_beta_mean",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["beta_mean"]),
        )
        self.register_buffer(
            "s_overdispersion_factor_alpha_sd",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["alpha_sd"]),
        )
        self.register_buffer(
            "s_overdispersion_factor_beta_sd",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["beta_sd"]),
        )
        
        self.register_buffer(
            "detection_gi_prior_alpha",
            torch.tensor(self.detection_gi_prior["alpha"]),
        )
        self.register_buffer(
            "detection_gi_prior_beta",
            torch.tensor(self.detection_gi_prior["alpha"] / self.detection_gi_prior["mean"]),
        )
        
        self.register_buffer(
            "detection_i_prior_alpha",
            torch.tensor(self.detection_i_prior["alpha"]),
        )
        self.register_buffer(
            "detection_i_prior_beta",
            torch.tensor(self.detection_i_prior["alpha"] / self.detection_i_prior["mean"]),
        )
        
        self.register_buffer(
            "Tmax_mean",
            torch.tensor(Tmax_prior["mean"]),
        )
             
        self.register_buffer(
            "Tmax_sd",
            torch.tensor(Tmax_prior["sd"]),
        )

        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_beta"]),
        )

        self.register_buffer(
            "stochastic_v_ag_hyp_prior_alpha",
            torch.tensor(self.stochastic_v_ag_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "stochastic_v_ag_hyp_prior_beta",
            torch.tensor(self.stochastic_v_ag_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )
        
        self.register_buffer(
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        
        self.register_buffer("one", torch.tensor(1.))


    @beartype
    def __repr__(self) -> str:
        return (
            f"\nKnnModel(\n"
            f"\tnum_cells={self.num_cells}, \n"
            f"\tnum_genes={self.num_genes}, \n"
            f")\n"
        )

    @beartype
    def forward(self,
                u_obs: torch.Tensor,
                s_obs: torch.Tensor,
                N_cn: torch.Tensor,
                ind_x: torch.Tensor,
                batch_index: torch.Tensor):
        """
        Defines the forward model, which computes the unspliced (u) and spliced
        (s) RNA expression levels given the observations and model parameters.

        Args:
            u_obs (Optional[torch.Tensor], optional): Observed unspliced RNA expression. Default is None.
            s_obs (Optional[torch.Tensor], optional): Observed spliced RNA expression. Default is None.
            ind_x (Optional[torch.Tensor], optional): Indices for the cells.
            batch_index (Optional[torch.Tensor], optional): Experimental batch index of cells.

        Returns:    

        Examples:
            >>> .
        """
        
        batch_size = len(ind_x)
        obs2sample = one_hot(batch_index, self.n_batch)        
        obs_plate = self.create_plates(u_obs, s_obs, ind_x, batch_index)
        k = N_cn.shape[1]   
        
        # ============= Expression Model =============== #
        T_max = pyro.sample('Tmax', dist.Gamma(G_a(self.Tmax_mean, self.Tmax_sd), G_b(self.Tmax_mean, self.Tmax_sd)))
        t_c_loc = pyro.sample('t_c_loc', dist.Gamma(self.one, self.one/0.5))
        t_c_scale = pyro.sample('t_c_scale', dist.Gamma(self.one, self.one/0.25))
        with obs_plate:
            t_c = pyro.sample('t_c', dist.Normal(t_c_loc, t_c_scale).expand([batch_size, 1, 1]))
            T_c = pyro.deterministic('T_c', t_c*T_max)
        
        # Time difference between neighbors:
        delta_cn = T_c.unsqueeze(-1) - T_c[N_cn, :]

        # Counts in each cell:
        mu0_cg = pyro.sample('mu0_cg', dist.Gamma(self.one*5.0, self.one*1.0).expand([batch_size, n_genes, 2]))

        # Weight of each nearest neighbor:    
        wdash0_nc = pyro.sample('wdash0_nc', dist.Gamma(self.one*0.1, self.one*10.0).expand([1,batch_size]))
        wdash5_nc = pyro.sample('wdash_nc', dist.Gamma(self.one*10.0, self.one*10.0).expand([k-1,batch_size]))
        wdash_nc = torch.concat([wdash0_nc, wdash5_nc], axis = 0)
        w_nc = wdash_nc/torch.sum(wdash_nc, axis = 0)

        # Predicted counts from each neighbor:
        y = (mu0_cg[...,0], mu0_cg[...,1])
        dy_cn = torch.stack(vector_field_1(0.0,y,[regulatory_function_1]), axis = -1)[N_cn,...]
        muhat_cg = torch.stack(y, axis = -1) + torch.sum((w_nc.T.unsqueeze(-1).unsqueeze(-1) * delta_cn * dy_cn), axis = 1)/k      
        
        # Initial conditions and predicted counts need to match up:
        with obs_plate:
            pyro.sample("data_target", dist.Gamma(muhat_cg, self.one, obs=mu0_cg))
        
        # ============= Measurement Model =============== #
        # Cell specific relative detection efficiency with hierarchical prior across batches:
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Beta(
                self.ones * self.detection_mean_hyp_prior_alpha,
                self.ones * self.detection_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        detection_hyp_prior_alpha = pyro.deterministic(
            "detection_hyp_prior_alpha",
            self.detection_hyp_prior_alpha,
        )

        beta = detection_hyp_prior_alpha / (obs2sample @ detection_mean_y_e)
        with obs_plate:
            detection_y_c = pyro.sample(
                "detection_y_c",
                dist.Gamma(detection_hyp_prior_alpha.unsqueeze(dim=-1), beta.unsqueeze(dim=-1)),
            )  # (self.n_obs, 1)        
        
        # Global relative detection efficiency between spliced and unspliced counts
        detection_y_i = pyro.sample(
            "detection_y_i",
            dist.Gamma(
                self.ones * self.detection_i_prior_alpha,
                self.ones * self.detection_i_prior_alpha,
            )
            .expand([1, 1, 2]).to_event(3)
        )
        
        # Gene specific relative detection efficiency between spliced and unspliced counts
        detection_y_gi = pyro.sample(
            "detection_y_gi",
            dist.Gamma(
                self.ones * self.detection_gi_prior_alpha,
                self.ones * self.detection_gi_prior_alpha,
            )
            .expand([1, self.n_vars, 2])
            .to_event(3),
        )
        
        # Gene-specific additive component (Ambient RNA/ "Soup") #
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.gene_add_alpha_hyp_prior_alpha, self.gene_add_alpha_hyp_prior_beta).expand([2]).to_event(1),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1, 2])
            .to_event(3),
        ) 
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1, 2]).to_event(3),
        )
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)
        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars, 2])
            .to_event(3),
        )  

        # =====================Expected observed expression ======================= #
        with obs_plate:
            mu = pyro.deterministic('mu', (mu0_cg + torch.einsum('cbi,bgi->cgi', obs2sample.unsqueeze(dim=-1), s_g_gene_add)) * \
        detection_y_c * detection_y_i * detection_y_gi)
        
        # =====================DATA likelihood ======================= #
        with obs_plate:
            pyro.sample("data_target", dist.Poisson(rate = mu,
                                                    obs=torch.stack([u_obs, s_obs], axis = 2)))