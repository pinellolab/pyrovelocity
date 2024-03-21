from os import PathLike
from typing import Dict

import numpy as np
from beartype import beartype
from matplotlib import pyplot as plt
from matplotlib.figure import FigureBase

from pyrovelocity.logging import configure_logging


logger = configure_logging(__name__)

__all__ = ["plot_t0_selection"]


@beartype
def plot_t0_selection(
    posterior_samples: Dict[str, np.ndarray],
    t0_selection_plot: PathLike | str,
) -> FigureBase:
    """
    Plot the t0, switching time, and cell time posterior samples.

    Args:
        posterior_samples (Dict[str, np.ndarray]): Dictionary of posterior samples.
        t0_selection_plot (PathLike | str): Path to save the plot.

    Returns:
        FigureBase: Matplotlib figure object.
    """
    fig, ax = plt.subplots(5, 6)
    fig.set_size_inches(26, 24)
    ax = ax.flatten()

    posterior_samples_keys_to_check = ["t0", "switching", "cell_time"]

    num_samples_list = [
        posterior_samples[key].shape[0]
        for key in posterior_samples_keys_to_check
    ]
    assert (
        len(set(num_samples_list)) == 1
    ), f"The number of samples is not equal across keys: {posterior_samples_keys_to_check}"

    num_samples = num_samples_list[0]

    for sample in range(num_samples):
        t0_sample = posterior_samples["t0"][sample]
        switching_sample = posterior_samples["switching"][sample]
        cell_time_sample = posterior_samples["cell_time"][sample]
        ax[sample].scatter(
            t0_sample.flatten(),
            2 * np.ones(t0_sample.shape[-1]),
            s=1,
            c="red",
            label="t0",
        )
        ax[sample].scatter(
            switching_sample.flatten(),
            3 * np.ones(t0_sample.shape[-1]),
            s=1,
            c="purple",
            label="switching",
        )
        ax[sample].scatter(
            cell_time_sample.flatten(),
            np.ones(cell_time_sample.shape[0]),
            s=1,
            c="blue",
            label="shared time",
        )
        ax[sample].set_ylim(-0.5, 4)
        if sample == 28:
            ax[sample].legend(loc="best", bbox_to_anchor=(0.5, 0.0, 0.5, 0.5))
    ax[-1].hist(t0_sample.flatten(), bins=200, color="red", alpha=0.3)
    ax[-1].hist(cell_time_sample.flatten(), bins=500, color="blue", alpha=0.3)

    fig.savefig(
        t0_selection_plot,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )

    return fig


# from typing import Dict
# from typing import List

# import anndata
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pyro
# import scvelo as scv
# import seaborn as sns
# import torch
# from adjustText import adjust_text
# from anndata import AnnData
# from matplotlib import cm
# from matplotlib import gridspec
# from matplotlib.colors import Normalize
# from matplotlib.figure import Figure
# from matplotlib.ticker import MaxNLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from numpy import ndarray
# from scipy.sparse import issparse
# from scipy.stats import spearmanr
# from scvelo.plotting.velocity_embedding_grid import default_arrow

# from pyrovelocity.analyze import compute_mean_vector_field
# from pyrovelocity.analyze import compute_volcano_data
# from pyrovelocity.models import mrna_dynamics


# def plot_evaluate_dynamic_orig(adata, gene="Cpe", velocity=None, ax=None):
#     # compute dynamics
#     alpha, beta, gamma, scaling, t_ = (
#         torch.tensor(adata.var.loc[gene, "fit_alpha"]),
#         torch.tensor(adata.var.loc[gene, "fit_beta"]),
#         torch.tensor(adata.var.loc[gene, "fit_gamma"]),
#         torch.tensor(adata.var.loc[gene, "fit_scaling"]),
#         torch.tensor(adata.var.loc[gene, "fit_t_"]),
#     )

#     beta_scale = beta * scaling

#     u0, s0 = adata.var.loc[gene, "fit_u0"], adata.var.loc[gene, "fit_s0"]
#     # t = torch.tensor(adata[:, gene].layers['fit_t'][:, 0]).sort()[0]
#     t = torch.tensor(adata[:, gene].layers["fit_t"][:, 0])
#     u_inf, s_inf = mrna_dynamics(t_, u0, s0, alpha, beta_scale, gamma)

#     state = (t < t_).int()
#     tau = t * state + (t - t_) * (1 - state)
#     u0_vec = u0 * state + u_inf * (1 - state)
#     s0_vec = s0 * state + s_inf * (1 - state)

#     alpha_ = 0
#     alpha_vec = alpha * state + alpha_ * (1 - state)

#     ut, st = mrna_dynamics(tau, u0_vec, s0_vec, alpha_vec, beta_scale, gamma)
#     ut = ut * scaling + u0
#     st = st + s0

#     xnew = torch.linspace(torch.min(st), torch.max(st))
#     ynew = gamma / beta * (xnew - torch.min(xnew)) + torch.min(ut)

#     if ax is None:
#         fig, ax = plt.subplots()
#         scv.pl.scatter(adata, gene, color=["clusters"], ax=ax, show=False)
#     else:
#         ax.scatter(
#             st.detach().numpy(),
#             ut.detach().numpy(),
#             linestyle="-",
#             linewidth=5,
#             alpha=0.3,
#         )
#         ax.plot(
#             xnew.detach().numpy(),
#             ynew.detach().numpy(),
#             color="b",
#             linestyle="--",
#             linewidth=5,
#         )

#     if velocity is None:
#         print(
#             "scvelo %s mse loss:" % gene,
#             mse_loss_sum(
#                 ut,
#                 st,
#                 adata[:, gene].layers["Mu"].toarray()[:, 0],
#                 adata[:, gene].layers["Ms"].toarray()[:, 0],
#             ),
#         )
#     else:
#         print(
#             "scvelo %s mse loss:" % gene,
#             mse_loss_sum(
#                 ut[velocity.weight],
#                 st[velocity.weight],
#                 adata[:, gene].layers["Mu"].toarray()[velocity.weight, 0],
#                 adata[:, gene].layers["Ms"].toarray()[velocity.weight, 0],
#             ),
#         )
#     return alpha_vec, ut, st, xnew, ynew


# def plot_dynamic_pyro(
#     adata,
#     gene,
#     losses,
#     summary,
#     velocity,
#     fix_param_list,
#     alpha,
#     beta,
#     gamma,
#     scale,
#     t_,
#     t,
# ):
#     alpha = (
#         torch.tensor(alpha)
#         if fix_param_list[0] == 1
#         else pyro.param("AutoDelta.alpha_sample")
#     )
#     beta = (
#         torch.tensor(beta)
#         if fix_param_list[1] == 1
#         else pyro.param("AutoDelta.beta_sample")
#     )
#     gamma = (
#         torch.tensor(gamma)
#         if fix_param_list[2] == 1
#         else pyro.param("AutoDelta.gamma_sample")
#     )
#     scale = (
#         torch.tensor(scale)
#         if fix_param_list[3] == 1
#         else pyro.param("AutoDelta.scale_sample")
#     )
#     t_ = (
#         torch.tensor(t_)
#         if fix_param_list[4] == 1
#         else pyro.param("AutoDelta.switching_sample")
#     )

#     if fix_param_list[5] == 0:
#         t = pyro.param("AutoDelta.latent_time")
#     else:
#         t = torch.tensor(t)
#     fig, ax = plt.subplots(1, 3)
#     fig.set_size_inches(16, 4)
#     ax[1].scatter(
#         adata[:, gene].layers["fit_t"].toarray()[velocity.weight, 0],
#         t.data.cpu().numpy(),
#     )

#     t = (t.sort()[0].max() + 1).int()
#     t = torch.linspace(0.0, t, 500)

#     beta_scale = beta * scale
#     u0, s0 = torch.tensor(0.0), torch.tensor(0.0)
#     u_inf, s_inf = mrna_dynamics(t_, u0, s0, alpha, beta_scale, gamma)

#     state = (t < t_).int()
#     tau = t * state + (t - t_) * (1 - state)
#     u0_vec = u0 * state + u_inf * (1 - state)
#     s0_vec = s0 * state + s_inf * (1 - state)

#     alpha_ = 0.0
#     alpha_vec = alpha * state + alpha_ * (1 - state)
#     ut, st = mrna_dynamics(tau, u0_vec, s0_vec, alpha_vec, beta_scale, gamma)
#     ut = ut * scale + u0
#     st = st + s0
#     xnew = torch.linspace(
#         torch.tensor(st.min().detach().numpy()),
#         torch.tensor(st.max().detach().numpy()),
#         50,
#     )
#     ynew = gamma / beta * (xnew - torch.min(xnew)) + torch.min(ut)

#     ax[0].plot(losses)
#     ax[0].set_yscale("log")
#     ax[0].set_title("ELBO")
#     ax[0].set_xlabel("step")
#     ax[0].set_ylabel("loss")

#     scv.pl.scatter(adata, gene, color=["clusters"], ax=ax[2], show=False)
#     ax[2].scatter(
#         summary["x_obs"]["mean"][:, 1],
#         summary["x_obs"]["mean"][:, 0],
#         alpha=0.5,
#         s=5,
#         color="r",
#     )
#     # ax[2].scatter(summary['x_obs']['5%'][:, 1], summary['x_obs']['5%'][:, 0], alpha=0.2, color='r')
#     # ax[2].scatter(summary['x_obs']['95%'][:, 1], summary['x_obs']['95%'][:, 0], alpha=0.2, color='r')
#     ax[2].plot(
#         st.detach().numpy(),
#         ut.detach().numpy(),
#         linestyle="-",
#         linewidth=5,
#         color="g",
#     )
#     ax[2].plot(
#         xnew.detach().numpy(),
#         ynew.detach().numpy(),
#         color="g",
#         linestyle="--",
#         linewidth=5,
#     )
#     # ax[2].set_ylim(0, 3)
#     # ax[2].set_xlim(0, 16)
#     plot_evaluate_dynamic_orig(adata, gene, velocity, ax[2])

#     print(
#         "pyro model mse loss",
#         mse_loss_sum(
#             summary["x_obs"]["mean"][:, 0],
#             summary["x_obs"]["mean"][:, 1],
#             velocity.x[:, 0],
#             velocity.x[:, 1],
#         ),
#     )
#     return alpha_vec, ut, st, xnew, ynew


# def plot_multigenes_dynamical(
#     summary,
#     alpha,
#     beta,
#     gamma,
#     t_,
#     t,
#     adata,
#     gene="Cpe",
#     scale=None,
#     ax=None,
#     raw=False,
# ):
#     pass

#     # softplus operation as pyro
#     # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
#     # t_ = torch.log(1+torch.exp(-np.abs(t_))) + torch.maximum(t_,
#     #     torch.zeros(t_.shape))

#     t = t.sort()[0].max().int()
#     t = torch.linspace(0.0, t, 500)
#     u0, s0 = torch.tensor(0.0), torch.tensor(0.0)
#     # u0, s0 = pyro.param("u0"), pyro.param("s0")
#     u_inf, s_inf = mrna_dynamics(t_, u0, s0, alpha, beta, gamma)
#     state = (t < t_).int()
#     tau = t * state + (t - t_) * (1 - state)

#     # tau = torch.log(1+torch.exp(-np.abs(tau))) + torch.maximum(tau,
#     #    torch.zeros(tau.shape))

#     u0_vec = u0 * state + u_inf * (1 - state)
#     s0_vec = s0 * state + s_inf * (1 - state)
#     alpha_ = 0.0
#     alpha_vec = alpha * state + alpha_ * (1 - state)
#     ut, st = mrna_dynamics(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
#     if scale is None:
#         ut = ut + u0
#     else:
#         ut = ut * scale + u0
#     st = st + s0
#     xnew = torch.linspace(
#         torch.tensor(st.min().detach().numpy()),
#         torch.tensor(st.max().detach().numpy()),
#         50,
#     )
#     if scale is not None:
#         ynew = (gamma / beta * (xnew - torch.min(xnew))) * scale + torch.min(
#             ut * scale + u0
#         )
#     else:
#         ynew = gamma / beta * (xnew - torch.min(xnew)) + torch.min(ut)

#     if ax is None:
#         fig, ax = plt.subplots()
#     try:
#         if raw:
#             scv.pl.scatter(
#                 adata,
#                 gene,
#                 x="spliced",
#                 y="unspliced",
#                 color=["clusters"],
#                 ax=ax,
#                 show=False,
#             )
#         else:
#             scv.pl.scatter(adata, gene, color=["clusters"], ax=ax, show=False)
#     except:
#         if raw:
#             scv.pl.scatter(
#                 adata,
#                 gene,
#                 x="spliced",
#                 y="unspliced",
#                 color=["Clusters"],
#                 ax=ax,
#                 show=False,
#             )
#         else:
#             scv.pl.scatter(adata, gene, color=["Clusters"], ax=ax, show=False)
#     ax.plot(
#         st.detach().numpy(),
#         ut.detach().numpy(),
#         linestyle="-",
#         linewidth=2.5,
#         color="red",
#         label="Pyro-Velocity",
#     )
#     ax.plot(
#         xnew.detach().numpy(),
#         ynew.detach().numpy(),
#         color="red",
#         linestyle="--",
#         linewidth=2.5,
#         alpha=0.4,
#     )
#     if summary is not None:
#         ax.scatter(
#             summary["x_obs"]["mean"][:, 1],
#             summary["x_obs"]["mean"][:, 0],
#             alpha=0.5,
#             color="red",
#         )
# def denoised_umap(posterior_samples, adata, cell_state="state_info"):
#     pass

#     import sklearn
#     import umap
#     from sklearn.pipeline import Pipeline

#     projection = [
#         ("PCA", sklearn.decomposition.PCA(random_state=99, n_components=50)),
#         ("UMAP", umap.UMAP(random_state=99, n_components=2)),
#     ]
#     pipelines = Pipeline(projection)
#     fig, ax = plt.subplots(2, 2)
#     fig.set_size_inches(9, 9)
#     expression = [posterior_samples["st"].mean(0)]
#     pipelines.fit(expression[0])
#     umap_orig = pipelines.transform(expression[0])
#     adata.obsm["X_umap1"] = umap_orig
#     scv.pl.scatter(adata, basis="umap1", ax=ax[0][0], show=False)

#     joint_pcs = pipelines.steps[0][1].transform(expression[0])
#     adata.obsm["X_pyropca"] = joint_pcs
#     scv.pp.neighbors(adata, use_rep="pyropca")
#     adata.layers["spliced_pyro"] = posterior_samples["st"].mean(0)
#     if "u_scale" in posterior_samples:
#         adata.layers["velocity_pyro"] = (
#             posterior_samples["ut"]
#             * posterior_samples["beta"]
#             / (posterior_samples["u_scale"] / posterior_samples["s_scale"])
#             - posterior_samples["st"] * posterior_samples["gamma"]
#         ).mean(0)
#     else:
#         adata.layers["velocity_pyro"] = (
#             posterior_samples["ut"] * posterior_samples["beta"]
#             - posterior_samples["st"] * posterior_samples["gamma"]
#         ).mean(0)
#     scv.tl.velocity_graph(adata, vkey="velocity_pyro", xkey="spliced_pyro")
#     scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis="umap1")
#     scv.pl.velocity_embedding_grid(
#         adata,
#         basis="umap1",
#         vkey="velocity_pyro",
#         density=0.5,
#         scale=0.25,
#         arrow_size=3,
#         color=cell_state,
#         ax=ax[0][1],
#         show=False,
#     )
#     adata.obsm["X_umap1"] = umap_orig

#     expression = [
#         np.hstack(
#             [posterior_samples["st"].mean(0), posterior_samples["ut"].mean(0)]
#         )
#     ]
#     pipelines.fit(expression[0])
#     umap_orig = pipelines.transform(expression[0])
#     adata.obsm["X_umap2"] = umap_orig
#     scv.pl.scatter(adata, basis="umap2", ax=ax[1][0], show=False)
#     joint_pcs = pipelines.steps[0][1].transform(expression[0])
#     adata.obsm["X_pyropca"] = joint_pcs
#     scv.pp.neighbors(adata, use_rep="pyropca")
#     adata.layers["spliced_pyro"] = posterior_samples["st"].mean(0)
#     if "u_scale" in posterior_samples:
#         adata.layers["velocity_pyro"] = (
#             posterior_samples["ut"]
#             * posterior_samples["beta"]
#             / (posterior_samples["u_scale"] / posterior_samples["s_scale"])
#             - posterior_samples["st"] * posterior_samples["gamma"]
#         ).mean(0)
#     else:
#         adata.layers["velocity_pyro"] = (
#             posterior_samples["ut"] * posterior_samples["beta"]
#             - posterior_samples["st"] * posterior_samples["gamma"]
#         ).mean(0)
#     scv.tl.velocity_graph(adata, vkey="velocity_pyro", xkey="spliced_pyro")
#     scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis="umap1")
#     scv.pl.velocity_embedding_grid(
#         adata,
#         basis="umap2",
#         vkey="velocity_pyro",
#         color=cell_state,
#         density=0.5,
#         scale=0.25,
#         arrow_size=3,
#         show=False,
#         ax=ax[1][1],
#     )

# def mse_loss_sum(u_model, s_model, u_data, s_data):
#     """
#     Computes the mean squared error loss sum between the model and data.

#     Args:
#         u_model (torch.Tensor): Predicted values of u from the model.
#         s_model (torch.Tensor): Predicted values of s from the model.
#         u_data (torch.Tensor): True values of u from the data.
#         s_data (torch.Tensor): True values of s from the data.

#     Returns:
#         torch.Tensor: Mean squared error loss sum.

#     Examples:
#         >>> import torch
#         >>> u_model = torch.tensor([0.5, 0.6])
#         >>> s_model = torch.tensor([0.7, 0.8])
#         >>> u_data = torch.tensor([0.4, 0.5])
#         >>> s_data = torch.tensor([0.6, 0.7])
#         >>> mse_loss_sum(u_model, s_model, u_data, s_data)
#         tensor(0.0200)
#     """
#     return ((u_model - u_data) ** 2 + (s_model - s_data) ** 2).mean(0)
