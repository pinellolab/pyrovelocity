from typing import Optional, Tuple, Union

import pyro
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
    "LogNormalModel",
    "VelocityModelAuto",
    "MultiVelocityModelAuto",
]

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
                ind_x: torch.Tensor,
                batch_index: torch.Tensor)
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
        
        # ===================== Kinetic Rates ======================= #
        # Splicing rate:
        splicing_alpha = pyro.sample('splicing_alpha',
                              dist.Gamma(self.splicing_rate_alpha_hyp_prior_alpha,
                              self.splicing_rate_alpha_hyp_prior_alpha/self.splicing_rate_alpha_hyp_prior_mean))
        splicing_mean = pyro.sample('splicing_mean',
                              dist.Gamma(self.splicing_rate_mean_hyp_prior_alpha,
                              self.splicing_rate_mean_hyp_prior_alpha/self.splicing_rate_mean_hyp_prior_mean))
        beta_g = pyro.sample('beta_g', dist.Gamma(splicing_alpha, splicing_alpha/splicing_mean).expand([1,self.n_vars]).to_event(2))
        # Degredation rate:
        degredation_alpha = pyro.sample('degredation_alpha',
                              dist.Gamma(self.degredation_rate_alpha_hyp_prior_alpha,
                              self.degredation_rate_alpha_hyp_prior_alpha/self.degredation_rate_alpha_hyp_prior_mean))
        degredation_alpha = degredation_alpha + 0.001
        degredation_mean = pyro.sample('degredation_mean',
                              dist.Gamma(self.degredation_rate_mean_hyp_prior_alpha,
                              self.degredation_rate_mean_hyp_prior_alpha/self.degredation_rate_mean_hyp_prior_mean))
        gamma_g = pyro.sample('gamma_g', dist.Gamma(degredation_alpha, degredation_alpha/degredation_mean).expand([1,self.n_vars]).to_event(2))
        # Transcription rate contribution of each module:
        factor_level_g = pyro.sample(
            "factor_level_g",
            dist.Gamma(self.factor_prior_alpha, self.factor_prior_beta)
            .expand([1, self.n_vars])
            .to_event(2)
        )
        g_fg = pyro.sample( # (g_fg corresponds to module's spliced counts in steady state)
            "g_fg",
            dist.Gamma(
                self.factor_states_per_gene / self.n_factors_torch,
                self.ones / factor_level_g,
            )
            .expand([self.n_modules, self.n_vars])
            .to_event(2)
        )
        A_mgON = pyro.deterministic('A_mgON', g_fg*gamma_g) # (transform from spliced counts to transcription rate)
        A_mgOFF = self.alpha_OFFg        
        # Activation and Deactivation rate:
        lam_mu = pyro.sample('lam_mu', dist.Gamma(G_a(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd),
                                            G_b(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd)))
        lam_sd = pyro.sample('lam_sd', dist.Gamma(G_a(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd),
                                            G_b(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd)))
        lam_m_mu = pyro.sample('lam_m_mu', dist.Gamma(G_a(lam_mu, lam_sd),
                                            G_b(lam_mu, lam_sd)).expand([self.n_modules, 1, 1]).to_event(3))
        lam_mi = pyro.sample('lam_mi', dist.Gamma(G_a(lam_m_mu, lam_m_mu*0.05),
                                            G_b(lam_m_mu, lam_m_mu*0.05)).expand([self.n_modules, 1, 2]).to_event(3))
        
        # =====================Time======================= #
        # Global time for each cell:
        T_max = pyro.sample('Tmax', dist.Gamma(G_a(self.Tmax_mean, self.Tmax_sd), G_b(self.Tmax_mean, self.Tmax_sd)))
        t_c_loc = pyro.sample('t_c_loc', dist.Gamma(self.one, self.one/0.5))
        t_c_scale = pyro.sample('t_c_scale', dist.Gamma(self.one, self.one/0.25))
        with obs_plate:
            t_c = pyro.sample('t_c', dist.Normal(t_c_loc, t_c_scale).expand([batch_size, 1, 1]))
            T_c = pyro.deterministic('T_c', t_c*T_max)
        # Global switch on time for each gene:
#         t_mON = pyro.sample('t_mON', dist.Uniform(self.zero, self.one).expand([1, 1, self.n_modules]).to_event(2))
        t_delta = pyro.sample('t_delta', dist.Gamma(self.one*20, self.one * 20 *self.n_modules_torch).
                              expand([self.n_modules]).to_event(1))
        t_mON = torch.cumsum(torch.concat([self.zero.unsqueeze(0), t_delta[:-1]]), dim = 0).unsqueeze(0).unsqueeze(0)
        T_mON = pyro.deterministic('T_mON', T_max*t_mON)
        # Global switch off time for each gene:
        t_mOFF = pyro.sample('t_mOFF', dist.Exponential(self.n_modules_torch).expand([1, 1, self.n_modules]).to_event(2))
        T_mOFF = pyro.deterministic('T_mOFF', T_mON + T_max*t_mOFF)
        
        # =========== Mean expression according to RNAvelocity model ======================= #
        mu_total = torch.stack([self.zeros[idx,...], self.zeros[idx,...]], axis = -1)
        for m in range(self.n_modules):
            mu_total += mu_mRNA_continousAlpha_globalTime_twoStates(
                A_mgON[m,:], A_mgOFF, beta_g, gamma_g, lam_mi[m,...], T_c[:,:,0], T_mON[:,:,m], T_mOFF[:,:,m], self.zeros[ind_x,...])
        with obs_plate:
            mu_expression = pyro.deterministic('mu_expression', mu_total)
        
        # =============Detection efficiency of spliced and unspliced counts =============== #
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
        
        # =======Gene-specific additive component (Ambient RNA/ "Soup") for spliced and unspliced counts ====== #
        # Independently sampled for spliced and unspliced counts:
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

        # =========Gene-specific overdispersion of spliced and unspliced counts ============== #
        # Overdispersion of unspliced counts:
        stochastic_v_ag_hyp = pyro.sample(
        "stochastic_v_ag_hyp",
        dist.Gamma(
            self.stochastic_v_ag_hyp_prior_alpha,
            self.stochastic_v_ag_hyp_prior_beta,
        ).expand([1, 2]).to_event(2))
        stochastic_v_ag_hyp = stochastic_v_ag_hyp + 0.001
        stochastic_v_ag_inv = pyro.sample(
            "stochastic_v_ag_inv",
            dist.Exponential(stochastic_v_ag_hyp)
            .expand([1, self.n_vars, 2]).to_event(3),
        ) 
        stochastic_v_ag = (self.ones / stochastic_v_ag_inv.pow(2))        

        # =====================Expected expression ======================= #
        with obs_plate:
            mu = pyro.deterministic('mu', (mu_expression + torch.einsum('cbi,bgi->cgi', obs2sample.unsqueeze(dim=-1), s_g_gene_add)) * \
        detection_y_c * detection_y_i * detection_y_gi)
        
        # =====================DATA likelihood ======================= #
        with obs_plate:
            pyro.sample("data_target", dist.GammaPoisson(concentration= stochastic_v_ag,
                       rate= stochastic_v_ag / mu), obs=torch.stack([u_obs, s_obs], axis = 2))