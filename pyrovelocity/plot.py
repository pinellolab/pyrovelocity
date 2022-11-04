from typing import Dict
from typing import Tuple
from typing import List

import matplotlib
import matplotlib.cm as cm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import scvelo as scv
import seaborn as sns
import torch
from anndata import AnnData
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from scipy.stats import spearmanr

from pyrovelocity.cytotrace import compute_similarity2

from .utils import mRNA
from .utils import mse_loss_sum


def plot_evaluate_dynamic_orig(adata, gene="Cpe", velocity=None, ax=None):
    # compute dynamics
    alpha, beta, gamma, scaling, t_ = (
        torch.tensor(adata.var.loc[gene, "fit_alpha"]),
        torch.tensor(adata.var.loc[gene, "fit_beta"]),
        torch.tensor(adata.var.loc[gene, "fit_gamma"]),
        torch.tensor(adata.var.loc[gene, "fit_scaling"]),
        torch.tensor(adata.var.loc[gene, "fit_t_"]),
    )

    beta_scale = beta * scaling

    u0, s0 = adata.var.loc[gene, "fit_u0"], adata.var.loc[gene, "fit_s0"]
    # t = torch.tensor(adata[:, gene].layers['fit_t'][:, 0]).sort()[0]
    t = torch.tensor(adata[:, gene].layers["fit_t"][:, 0])
    u_inf, s_inf = mRNA(t_, u0, s0, alpha, beta_scale, gamma)

    state = (t < t_).int()
    tau = t * state + (t - t_) * (1 - state)
    u0_vec = u0 * state + u_inf * (1 - state)
    s0_vec = s0 * state + s_inf * (1 - state)

    alpha_ = 0
    alpha_vec = alpha * state + alpha_ * (1 - state)

    ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta_scale, gamma)
    ut = ut * scaling + u0
    st = st + s0

    xnew = torch.linspace(torch.min(st), torch.max(st))
    ynew = gamma / beta * (xnew - torch.min(xnew)) + torch.min(ut)

    if ax is None:
        fig, ax = plt.subplots()
        scv.pl.scatter(adata, gene, color=["clusters"], ax=ax, show=False)
    else:
        ax.scatter(
            st.detach().numpy(),
            ut.detach().numpy(),
            linestyle="-",
            linewidth=5,
            alpha=0.3,
        )
        ax.plot(
            xnew.detach().numpy(),
            ynew.detach().numpy(),
            color="b",
            linestyle="--",
            linewidth=5,
        )

    if velocity is None:
        print(
            "scvelo %s mse loss:" % gene,
            mse_loss_sum(
                ut,
                st,
                adata[:, gene].layers["Mu"].toarray()[:, 0],
                adata[:, gene].layers["Ms"].toarray()[:, 0],
            ),
        )
    else:
        print(
            "scvelo %s mse loss:" % gene,
            mse_loss_sum(
                ut[velocity.weight],
                st[velocity.weight],
                adata[:, gene].layers["Mu"].toarray()[velocity.weight, 0],
                adata[:, gene].layers["Ms"].toarray()[velocity.weight, 0],
            ),
        )
    return alpha_vec, ut, st, xnew, ynew


def plot_dynamic_pyro(
    adata,
    gene,
    losses,
    summary,
    velocity,
    fix_param_list,
    alpha,
    beta,
    gamma,
    scale,
    t_,
    t,
):
    alpha = (
        torch.tensor(alpha)
        if fix_param_list[0] == 1
        else pyro.param("AutoDelta.alpha_sample")
    )
    beta = (
        torch.tensor(beta)
        if fix_param_list[1] == 1
        else pyro.param("AutoDelta.beta_sample")
    )
    gamma = (
        torch.tensor(gamma)
        if fix_param_list[2] == 1
        else pyro.param("AutoDelta.gamma_sample")
    )
    scale = (
        torch.tensor(scale)
        if fix_param_list[3] == 1
        else pyro.param("AutoDelta.scale_sample")
    )
    t_ = (
        torch.tensor(t_)
        if fix_param_list[4] == 1
        else pyro.param("AutoDelta.switching_sample")
    )

    if fix_param_list[5] == 0:
        t = pyro.param("AutoDelta.latent_time")
    else:
        t = torch.tensor(t)
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(16, 4)
    ax[1].scatter(
        adata[:, gene].layers["fit_t"].toarray()[velocity.weight, 0],
        t.data.cpu().numpy(),
    )

    t = (t.sort()[0].max() + 1).int()
    t = torch.linspace(0.0, t, 500)

    beta_scale = beta * scale
    u0, s0 = torch.tensor(0.0), torch.tensor(0.0)
    u_inf, s_inf = mRNA(t_, u0, s0, alpha, beta_scale, gamma)

    state = (t < t_).int()
    tau = t * state + (t - t_) * (1 - state)
    u0_vec = u0 * state + u_inf * (1 - state)
    s0_vec = s0 * state + s_inf * (1 - state)

    alpha_ = 0.0
    alpha_vec = alpha * state + alpha_ * (1 - state)
    ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta_scale, gamma)
    ut = ut * scale + u0
    st = st + s0
    xnew = torch.linspace(
        torch.tensor(st.min().detach().numpy()),
        torch.tensor(st.max().detach().numpy()),
        50,
    )
    ynew = gamma / beta * (xnew - torch.min(xnew)) + torch.min(ut)

    ax[0].plot(losses)
    ax[0].set_yscale("log")
    ax[0].set_title("ELBO")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("loss")

    scv.pl.scatter(adata, gene, color=["clusters"], ax=ax[2], show=False)
    ax[2].scatter(
        summary["x_obs"]["mean"][:, 1],
        summary["x_obs"]["mean"][:, 0],
        alpha=0.5,
        s=5,
        color="r",
    )
    # ax[2].scatter(summary['x_obs']['5%'][:, 1], summary['x_obs']['5%'][:, 0], alpha=0.2, color='r')
    # ax[2].scatter(summary['x_obs']['95%'][:, 1], summary['x_obs']['95%'][:, 0], alpha=0.2, color='r')
    ax[2].plot(
        st.detach().numpy(), ut.detach().numpy(), linestyle="-", linewidth=5, color="g"
    )
    ax[2].plot(
        xnew.detach().numpy(),
        ynew.detach().numpy(),
        color="g",
        linestyle="--",
        linewidth=5,
    )
    # ax[2].set_ylim(0, 3)
    # ax[2].set_xlim(0, 16)
    plot_evaluate_dynamic_orig(adata, gene, velocity, ax[2])

    print(
        "pyro model mse loss",
        mse_loss_sum(
            summary["x_obs"]["mean"][:, 0],
            summary["x_obs"]["mean"][:, 1],
            velocity.x[:, 0],
            velocity.x[:, 1],
        ),
    )
    return alpha_vec, ut, st, xnew, ynew


def plot_multigenes_dynamical(
    summary,
    alpha,
    beta,
    gamma,
    t_,
    t,
    adata,
    gene="Cpe",
    scale=None,
    ax=None,
    raw=False,
):
    pass

    # softplus operation as pyro
    # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    # t_ = torch.log(1+torch.exp(-np.abs(t_))) + torch.maximum(t_,
    #     torch.zeros(t_.shape))

    t = t.sort()[0].max().int()
    t = torch.linspace(0.0, t, 500)
    u0, s0 = torch.tensor(0.0), torch.tensor(0.0)
    # u0, s0 = pyro.param("u0"), pyro.param("s0")
    u_inf, s_inf = mRNA(t_, u0, s0, alpha, beta, gamma)
    state = (t < t_).int()
    tau = t * state + (t - t_) * (1 - state)

    # tau = torch.log(1+torch.exp(-np.abs(tau))) + torch.maximum(tau,
    #    torch.zeros(tau.shape))

    u0_vec = u0 * state + u_inf * (1 - state)
    s0_vec = s0 * state + s_inf * (1 - state)
    alpha_ = 0.0
    alpha_vec = alpha * state + alpha_ * (1 - state)
    ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
    if scale is None:
        ut = ut + u0
    else:
        ut = ut * scale + u0
    st = st + s0
    xnew = torch.linspace(
        torch.tensor(st.min().detach().numpy()),
        torch.tensor(st.max().detach().numpy()),
        50,
    )
    if scale is not None:
        ynew = (gamma / beta * (xnew - torch.min(xnew))) * scale + torch.min(
            ut * scale + u0
        )
    else:
        ynew = gamma / beta * (xnew - torch.min(xnew)) + torch.min(ut)

    if ax is None:
        fig, ax = plt.subplots()
    try:
        if raw:
            scv.pl.scatter(
                adata,
                gene,
                x="spliced",
                y="unspliced",
                color=["clusters"],
                ax=ax,
                show=False,
            )
        else:
            scv.pl.scatter(adata, gene, color=["clusters"], ax=ax, show=False)
    except:
        if raw:
            scv.pl.scatter(
                adata,
                gene,
                x="spliced",
                y="unspliced",
                color=["Clusters"],
                ax=ax,
                show=False,
            )
        else:
            scv.pl.scatter(adata, gene, color=["Clusters"], ax=ax, show=False)
    ax.plot(
        st.detach().numpy(),
        ut.detach().numpy(),
        linestyle="-",
        linewidth=2.5,
        color="red",
        label="Pyro-Velocity",
    )
    ax.plot(
        xnew.detach().numpy(),
        ynew.detach().numpy(),
        color="red",
        linestyle="--",
        linewidth=2.5,
        alpha=0.4,
    )
    if summary is not None:
        ax.scatter(
            summary["x_obs"]["mean"][:, 1],
            summary["x_obs"]["mean"][:, 0],
            alpha=0.5,
            color="red",
        )


def plot_posterior_time(
    pos, adata, ax=None, fig=None, basis="umap", addition=True, position="left", s=3
):
    if addition:
        sns.set_style("white")
        sns.set_context("paper", font_scale=1)
        matplotlib.rcParams.update({"font.size": 7})
        # celltime_cors = []
        # celltime_labels = []
        # for index in range(pos['cell_time'].shape[0]):
        #    celltime_cors.append(spearmanr(1-adata.obs.cytotrace, pos['cell_time'][index])[0])
        #    celltime_labels.append('Pyro-Velocity train data')
        # celltime_cors.append(spearmanr(1-adata.obs.cytotrace, adata.obs.latent_time)[0])
        # celltime_labels.append('scvelo')
        # sns.boxplot(x='label', y='correlation',
        #            data=pd.DataFrame({"correlation": celltime_cors, "label": celltime_labels}))
        # plt.tick_params(axis='x', rotation=90)
        # plt.ylabel("Correlation with Cytotrace")
        # plt.xlabel("")
        # plt.title("Benchmark shared cell time")
        # plt.ylim(0, 1)
        plt.figure()
        test_hist = plt.hist(pos["cell_time"].mean(0), bins=100, label="test")
        plt.xlabel("mean of cell time")
        plt.ylabel("frequency")
        plt.title("Histogram of cell time posterior samples")
        plt.legend()
    pos_mean_time = pos["cell_time"].mean(0)
    # scale to 0-1?
    adata.obs["cell_time"] = pos_mean_time / pos_mean_time.max()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(2.36, 2)

    # scv.pl.umap(adata, color='cell_time', show=False, color_map='inferno',
    #            ax=ax, title='PyroVelocity shared time')

    im = ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=s,
        alpha=0.4,
        c=adata.obs["cell_time"],
        cmap="inferno",
        linewidth=0,
    )
    set_colorbar(im, ax, labelsize=5, fig=fig, position=position)
    # ax.arrow(-19, -6, 0, 5, length_includes_head=True,
    #         head_width=1, head_length=1, color='black')
    ax.axis("off")
    print(spearmanr(adata.obs["cell_time"].values, 1 - adata.obs.cytotrace.values)[1])
    print(spearmanr(adata.obs["cell_time"].values, 1 - adata.obs.cytotrace.values))
    ax.set_title(
        "Pyro-Velocity shared time\ncorrelation: %.2f"
        % (spearmanr(adata.obs["cell_time"].values, 1 - adata.obs.cytotrace.values)[0]),
        fontsize=7,
    )
    # fig.tight_layout()
    # plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.85)
    # return fig


def mae_per_gene(pred_counts: ndarray, true_counts: ndarray) -> ndarray:
    """Computes mean average error between counts and predicted probabilities."""
    error = np.abs(true_counts - pred_counts).sum(-2)
    total = np.clip(true_counts.sum(-2), 1, np.inf)
    return -np.array(error / total)


def plot_gene_ranking(
    pos,
    adata,
    ax=None,
    time_correlation_with="s",
    selected_genes=None,
    assemble=False,
    data="correlation",
    negative=False,
    adjust_text=False,
):
    assert isinstance(pos, tuple) or isinstance(pos, list)
    assert isinstance(adata, list) or isinstance(adata, tuple)

    maes_list = []
    cors = []
    genes = []
    labels = []
    switching = []
    for p, ad, label in zip(pos, adata, ["train", "valid"]):
        for sample in range(30):
            maes_list.append(
                mae_per_gene(
                    p["s"][sample].squeeze(), ad.layers["raw_spliced"].toarray()
                )
            )
            df_genes_cors = compute_similarity2(
                p[time_correlation_with][sample].squeeze(),
                p["cell_time"][sample].squeeze().reshape(-1, 1),
            )
            cors.append(df_genes_cors[0])
            genes.append(ad.var_names.values)
            # if 'dt_switching' in p:
            #     if 't0' in p:
            #         switching.append((p['t0'] + p['dt_switching'])[sample].flatten())
            #     else:
            #         switching.append((p['dt_switching'])[sample].flatten())
            labels.append(["Poisson_%s" % label] * len(ad.var_names.values))

    # if 'dt_switching' in p:
    #     volcano_data = pd.DataFrame({"mean_mae":         np.hstack(maes_list),
    #                                  "label":            np.hstack(labels),
    #                                  "time_correlation": np.hstack(cors),
    #                                  "switching":        np.hstack(switching),
    #                                  "genes":            np.hstack(genes)})
    # volcano_data = volcano_data.groupby("genes").mean(['mean_mae', 'time_correlation', 'switching'])
    # else:
    volcano_data = pd.DataFrame(
        {
            "mean_mae": np.hstack(maes_list),
            "label": np.hstack(labels),
            "time_correlation": np.hstack(cors),
            "genes": np.hstack(genes),
        }
    )
    volcano_data = volcano_data.groupby("genes").mean(["mean_mae", "time_correlation"])

    volcano_data.loc[:, "mean_mae_rank"] = volcano_data.mean_mae.rank(ascending=False)
    volcano_data.loc[:, "time_correlation_rank"] = volcano_data.time_correlation.apply(
        abs
    ).rank(ascending=False)
    volcano_data.loc[:, "rank_product"] = (
        volcano_data.mean_mae_rank * volcano_data.time_correlation_rank
    )

    if selected_genes is None:
        # selected_genes = volcano_data.sort_values('rank_product', ascending=True).head(20)
        # genes = np.hstack([selected_genes.loc[selected_genes.loc[:, 'time_correlation'] > 0, :].index[:2],
        #                   selected_genes.loc[selected_genes.loc[:, 'time_correlation'] < 0, :].index[:2]])
        genes = (
            volcano_data.sort_values("mean_mae", ascending=False)
            .head(300)
            .sort_values("time_correlation", ascending=negative)
            .head(4)
            .index
        )
    else:
        genes = selected_genes
    fig = None
    volcano_data.loc[:, "selected genes"] = 0
    volcano_data.loc[genes, "selected genes"] = 1

    from adjustText import adjust_text

    if data == "correlation":
        sns.scatterplot(
            x="time_correlation",
            y="mean_mae",
            hue="selected genes",
            data=volcano_data,
            s=3,
            linewidth=0,
            ax=ax,
            legend=False,
            alpha=0.3,
        )
        ax.set_title("Pyro-Velocity genes", fontsize=7)
        ax.set_xlabel("shared time correlation\nwith spliced expression", fontsize=7)
        ax.set_ylabel("negative mean\nabsolute error", fontsize=7)
        sns.despine()
        ax.tick_params(labelsize=6)
        texts = []
        for i, g in enumerate(genes):
            ax.scatter(
                volcano_data.loc[g, :].time_correlation,
                volcano_data.loc[g, :].mean_mae,
                s=15,
                color="red",
                marker="*",
            )
            # texts.append(ax.text(volcano_data.loc[g, :].time_correlation-0.45, volcano_data.loc[g, :].mean_mae-0.22*i,
            texts.append(
                ax.text(
                    volcano_data.loc[g, :].time_correlation,
                    volcano_data.loc[g, :].mean_mae,
                    g,
                    fontsize=5,
                    color="black",
                    ha="center",
                    va="center",
                )
            )
            if not assemble:
                if i % 2 == 0:
                    offset = 10 + i * 5
                else:
                    offset = -10 - i * 5
                if i % 2 == 0:
                    offset_y = -10 + i * 5
                else:
                    offset_y = -10 + i * 5
                # ax.annotate(
                #     g,
                #     xy=(volcano_data.loc[g, :].time_correlation, volcano_data.loc[g, :].mean_mae), xytext=(offset, offset_y),
                #     textcoords='offset points', ha='right', va='bottom',
                #     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                #     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='black'))
                if not adjust_text:
                    adjust_text(
                        texts,
                        arrowprops=dict(arrowstyle="-", color="red", alpha=0.5),
                        ha="center",
                        va="bottom",
                        ax=ax,
                    )
                else:
                    adjust_text(
                        texts,
                        precision=0.001,
                        expand_text=(1.01, 1.05),
                        expand_points=(1.01, 1.05),
                        force_text=(0.01, 0.25),
                        force_points=(0.01, 0.25),
                        arrowprops=dict(arrowstyle="-", color="blue", alpha=0.6),
                        ax=ax,
                    )
    else:
        sns.scatterplot(
            x="switching",
            y="mean_mae",
            hue="selected genes",
            data=volcano_data,
            s=3,
            linewidth=0,
            ax=ax,
            legend=False,
            alpha=0.3,
        )
        ax.set_title("Pyro-Velocity genes", fontsize=7)
        ax.set_xlabel("gene switching time", fontsize=7)
        ax.set_ylabel("negative mean\nabsolute error", fontsize=7)
        sns.despine()

    return volcano_data, fig


def denoised_umap(pos, adata, cell_state="state_info"):
    pass

    import sklearn
    import umap
    from sklearn.pipeline import Pipeline

    projection = [
        ("PCA", sklearn.decomposition.PCA(random_state=99, n_components=50)),
        ("UMAP", umap.UMAP(random_state=99, n_components=2)),
    ]
    pipelines = Pipeline(projection)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(9, 9)
    expression = [pos["st"].mean(0)]
    pipelines.fit(expression[0])
    umap_orig = pipelines.transform(expression[0])
    adata.obsm["X_umap1"] = umap_orig
    scv.pl.scatter(adata, basis="umap1", ax=ax[0][0], show=False)

    joint_pcs = pipelines.steps[0][1].transform(expression[0])
    adata.obsm["X_pyropca"] = joint_pcs
    scv.pp.neighbors(adata, use_rep="pyropca")
    adata.layers["spliced_pyro"] = pos["st"].mean(0)
    if "u_scale" in pos:
        adata.layers["velocity_pyro"] = (
            pos["ut"] * pos["beta"] / (pos["u_scale"] / pos["s_scale"])
            - pos["st"] * pos["gamma"]
        ).mean(0)
    else:
        adata.layers["velocity_pyro"] = (
            pos["ut"] * pos["beta"] - pos["st"] * pos["gamma"]
        ).mean(0)
    scv.tl.velocity_graph(adata, vkey="velocity_pyro", xkey="spliced_pyro")
    scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis="umap1")
    scv.pl.velocity_embedding_grid(
        adata,
        basis="umap1",
        vkey="velocity_pyro",
        density=0.5,
        scale=0.25,
        arrow_size=3,
        color=cell_state,
        ax=ax[0][1],
        show=False,
    )
    adata.obsm["X_umap1"] = umap_orig

    expression = [np.hstack([pos["st"].mean(0), pos["ut"].mean(0)])]
    pipelines.fit(expression[0])
    umap_orig = pipelines.transform(expression[0])
    adata.obsm["X_umap2"] = umap_orig
    scv.pl.scatter(adata, basis="umap2", ax=ax[1][0], show=False)
    joint_pcs = pipelines.steps[0][1].transform(expression[0])
    adata.obsm["X_pyropca"] = joint_pcs
    scv.pp.neighbors(adata, use_rep="pyropca")
    adata.layers["spliced_pyro"] = pos["st"].mean(0)
    if "u_scale" in pos:
        adata.layers["velocity_pyro"] = (
            pos["ut"] * pos["beta"] / (pos["u_scale"] / pos["s_scale"])
            - pos["st"] * pos["gamma"]
        ).mean(0)
    else:
        adata.layers["velocity_pyro"] = (
            pos["ut"] * pos["beta"] - pos["st"] * pos["gamma"]
        ).mean(0)
    scv.tl.velocity_graph(adata, vkey="velocity_pyro", xkey="spliced_pyro")
    scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis="umap1")
    scv.pl.velocity_embedding_grid(
        adata,
        basis="umap2",
        vkey="velocity_pyro",
        color=cell_state,
        density=0.5,
        scale=0.25,
        arrow_size=3,
        show=False,
        ax=ax[1][1],
    )


def vector_field_uncertainty(
    adata: AnnData,
    pos: Dict[str, ndarray],
    basis: str = "tsne",
    n_jobs: int = 1,
    denoised: bool = False,
) -> Tuple[ndarray, ndarray, ndarray]:
    import numpy as np
    import sklearn
    import umap
    from astropy.stats import rayleightest
    from sklearn.pipeline import Pipeline

    # fig, ax = plt.subplots(10, 3)
    # fig.set_size_inches(16, 36)
    # ax = ax.flatten()
    v_map_all = []
    if ("u_scale" in pos) and ("s_scale" in pos):
        scale = pos["u_scale"] / pos["s_scale"]
    else:
        scale = 1

    if "beta_k" in pos:
        velocity_samples = (
            pos["ut"] * pos["beta_k"] / scale - pos["st"] * pos["gamma_k"]
        )
    else:
        velocity_samples = pos["beta"] * pos["ut"] / scale - pos["gamma"] * pos["st"]

    if denoised:
        projection = [
            ("PCA", sklearn.decomposition.PCA(random_state=99, n_components=50)),
            ("UMAP", umap.UMAP(random_state=99, n_components=2)),
        ]
        pipelines = Pipeline(projection)
        expression = [pos["st"].mean(0)]
        pipelines.fit(expression[0])
        umap_orig = pipelines.transform(expression[0])
        adata.obsm["X_umap1"] = umap_orig
        joint_pcs = pipelines.steps[0][1].transform(expression[0])
        adata.obsm["X_pyropca"] = joint_pcs
        scv.pp.neighbors(adata, use_rep="pyropca")
    else:
        scv.pp.neighbors(adata, use_rep="pca")
    ##scv.pp.neighbors(adata, use_rep=basis)

    for sample in range(30):
        adata.layers["spliced_pyro"] = pos["st"][sample]
        adata.layers["velocity_pyro"] = velocity_samples[sample]
        adata.var["velocity_genes"] = True
        scv.tl.velocity_graph(
            adata, vkey="velocity_pyro", xkey="spliced_pyro", n_jobs=n_jobs
        )
        scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis=basis)
        v_map_all.append(adata.obsm[f"velocity_pyro_{basis}"])
    v_map_all = np.stack(v_map_all)
    embeds_radian = np.arctan2(v_map_all[:, :, 1], v_map_all[:, :, 0])
    from statsmodels.stats.multitest import multipletests

    rayleightest_pval = rayleightest(embeds_radian, axis=-2)
    _, fdri, _, _ = multipletests(rayleightest_pval, method="fdr_bh")
    return v_map_all, embeds_radian, fdri


def get_posterior_sample_angle_uncertainty(posterior_angles):
    from astropy import units as u
    from astropy.stats import circstd

    x_values = np.arange(360)
    y_values = []
    n_samples = 100
    method = "angular"
    for i in x_values:
        datac = np.linspace(0, i + 1, n_samples) * u.deg
        y_values.append(circstd(datac, method=method))
    y_values = np.array(y_values)
    angle_std = circstd(posterior_angles * u.deg, method="angular", axis=0)
    y_values = y_values.reshape(-1, 1)
    return x_values[np.argmin(np.abs(y_values - angle_std), axis=0)]


def plot_vector_field_uncertain(
    adata,
    embed_mean,
    embeds_radian,
    fig=None,
    cbar=True,
    basis="umap",
    scale=0.002,
    cbar_pos=[0.22, 0.28, 0.5, 0.05],
    p_mass_min=3.5,
    only_grid=False,
    ax=None,
    autoscale=False,
    density=0.3,
    arrow_size=5,
):
    from scvelo.plotting.velocity_embedding_grid import default_arrow

    if not only_grid:
        if ax is None:
            ax = fig.subplots(1, 2)
        dot_size = 1
        plt.rcParams["image.cmap"] = "winter"
        # norm = Normalize()
        # norm.autoscale(fdri_grids)
        # colormap = cm.inferno

        adata.obs["uncertain"] = get_posterior_sample_angle_uncertainty(
            embeds_radian / np.pi * 180
        )
        im = ax[0].scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            c=get_posterior_sample_angle_uncertainty(embeds_radian / np.pi * 180),
            s=dot_size,
            linewidth=0,
            vmin=0,
            vmax=360,
            cmap="winter",
        )
        ax[0].axis("off")
        ax[0].set_title("Single-cell\nvector field uncertainty ", fontsize=7)
        ax = ax[1]

    X_grid, V_grid, uncertain = project_grid_points(
        adata.obsm[f"X_{basis}"],
        embed_mean,
        get_posterior_sample_angle_uncertainty(embeds_radian / np.pi * 180),
        p_mass_min=p_mass_min,
        autoscale=autoscale,
        density=density,
    )
    # print(X_grid)
    # print(V_grid)
    # print(uncertain)

    # scale = None
    hl, hw, hal = default_arrow(arrow_size)
    print(hl, hw, hal)
    quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
    quiver_kwargs.update({"width": 0.001, "headlength": hl / 2})
    quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2})
    quiver_kwargs.update({"linewidth": 1, "zorder": 3})
    norm = Normalize()
    norm.autoscale(uncertain)
    colormap = cm.winter
    ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=1,
        linewidth=0,
        color="gray",
        alpha=0.2,
    )
    im = ax.quiver(
        X_grid[:, 0],
        X_grid[:, 1],
        V_grid[:, 0],
        V_grid[:, 1],
        # uncertain,
        color=colormap(norm(uncertain)),
        # ec=colormap(norm(uncertain)), # normalize to 0-1
        edgecolors=colormap(norm(uncertain)),
        # edgecolors='black',
        norm=Normalize(vmin=0, vmax=360),
        cmap="winter",
        scale=scale,
        **quiver_kwargs,
    )
    ax.set_title("Averaged\nvector field uncertainty ", fontsize=7)
    ax.axis("off")
    if cbar:
        from matplotlib.ticker import MaxNLocator

        # divider = make_axes_locatable(ax[1])
        # cax = divider.append_axes('bottom', size='5%', pad=0.08)
        # cbar = fig.colorbar(im, cax=cax, orientation="horizontal", shrink=0.6)
        # cbar.ax.set_xticks([0, 180, 360], [0, 180, 360])
        ##fig.colorbar(im, ax=ax, shrink=0.6, location='bottom')
        # cbar.ax.set_xlabel("Angle uncertainty", fontsize=6)
        cbar_ax = fig.add_axes(cbar_pos)
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", shrink=0.8)
        # cbar.tick_params(axis='x', labelsize=6)
        cbar.ax.set_xticks([0, 180, 360], [0, 180, 360], fontsize=6)
        cbar.ax.locator = MaxNLocator(nbins=2, integer=True)
        cbar.ax.set_xlabel("Angle uncertainty", fontsize=6)


def plot_mean_vector_field(
    pos,
    adata,
    ax,
    basis="umap",
    n_jobs=1,
    scale=0.2,
    density=0.4,
    spliced="spliced_pyro",
    raw=False,
):
    pass

    if basis == "umap":
        # projection = [ ('PCA', sklearn.decomposition.PCA(random_state=99, n_components=50)),
        #               ('UMAP', umap.UMAP(random_state=99, n_components=2)) ]
        # pipelines = Pipeline(projection)
        # expression = [pos['st'].mean(0)]
        # pipelines.fit(expression[0])
        # umap_orig = pipelines.transform(expression[0])
        # adata.obsm['X_umap1'] = umap_orig
        # joint_pcs = pipelines.steps[0][1].transform(expression[0])
        # adata.obsm['X_pyropca'] = joint_pcs
        # scv.pp.neighbors(adata, use_rep='pyropca')
        # pca = sklearn.decomposition.PCA(random_state=99, n_components=50)
        # expression = [pos['st'].mean(0)]
        # pca.fit(expression[0])
        # joint_pcs = pca.transform(expression[0])
        # adata.obsm['X_pyropca'] = joint_pcs
        scv.pp.neighbors(adata, use_rep="pca")
    else:
        # pca = sklearn.decomposition.PCA(random_state=99, n_components=50)
        # expression = [pos['st'].mean(0)]
        # pca.fit(expression[0])
        # joint_pcs = pca.transform(expression[0])
        # adata.obsm['X_pyropca'] = joint_pcs
        # scv.pp.neighbors(adata, use_rep='pyropca')
        scv.pp.neighbors(adata, use_rep="pca")
        ####scv.pp.neighbors(adata, use_rep=basis)

    adata.var["velocity_genes"] = True

    if spliced == "spliced_pyro":
        if raw:
            ut = pos["ut"]
            st = pos["st"]
            ut = ut / ut.sum(axis=-1, keepdims=True)
            st = st / st.sum(axis=-1, keepdims=True)
        else:
            ut = pos["ut"]
            st = pos["st"]
        # print(st.shape)
        adata.layers["spliced_pyro"] = st.mean(0).squeeze()
        # print(ut.shape)
        # print(pos['beta_k'].shape)
        # print(pos['beta'].shape)
        # if ('u_scale' in pos) and ('s_scale' in pos): # TODO: two scale for Normal distribution
        if "u_scale" in pos:  # only one scale for Poisson distribution
            adata.layers["velocity_pyro"] = (
                ut * pos["beta"] / pos["u_scale"] - st * pos["gamma"]
            ).mean(0)
        else:
            if "beta_k" in pos:
                adata.layers["velocity_pyro"] = (
                    (ut * pos["beta_k"] - pos["st"] * pos["gamma_k"]).mean(0).squeeze()
                )
            else:
                adata.layers["velocity_pyro"] = (
                    ut * pos["beta"] - pos["st"] * pos["gamma"]
                ).mean(0)
        scv.tl.velocity_graph(
            adata, vkey="velocity_pyro", xkey="spliced_pyro", n_jobs=n_jobs
        )
    elif spliced in ["Ms"]:
        ut = adata.layers["Mu"]
        st = adata.layers["Ms"]
        if ("u_scale" in pos) and ("s_scale" in pos):
            adata.layers["velocity_pyro"] = (
                ut * pos["beta"] / (pos["u_scale"] / pos["s_scale"]) - st * pos["gamma"]
            ).mean(0)
        else:
            adata.layers["velocity_pyro"] = (
                ut * pos["beta"] - pos["st"] * pos["gamma"]
            ).mean(0)
        scv.tl.velocity_graph(adata, vkey="velocity_pyro", xkey="Ms", n_jobs=n_jobs)
    elif spliced in ["spliced"]:
        ut = adata.layers["unspliced"]
        st = adata.layers["spliced"]
        if ("u_scale" in pos) and ("s_scale" in pos):
            adata.layers["velocity_pyro"] = (
                ut * pos["beta"] / (pos["u_scale"] / pos["s_scale"]) - st * pos["gamma"]
            ).mean(0)
        else:
            adata.layers["velocity_pyro"] = (
                ut * pos["beta"] - pos["st"] * pos["gamma"]
            ).mean(0)
        scv.tl.velocity_graph(
            adata, vkey="velocity_pyro", xkey="spliced", n_jobs=n_jobs
        )

    scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis=basis)
    scv.pl.velocity_embedding_grid(
        adata,
        basis=basis,
        vkey="velocity_pyro",
        linewidth=1,
        ax=ax,
        show=False,
        legend_loc="on data",
        density=density,
        scale=scale,
        arrow_size=3,
    )
    return adata.obsm[f"velocity_pyro_{basis}"]


# def project_grid_points(emb, velocity_emb, uncertain=None, p_mass_min=3.5, density=0.3):
def project_grid_points(
    emb, velocity_emb, uncertain=None, p_mass_min=1.0, density=0.3, autoscale=False
):
    from scipy.stats import norm as normal
    from scvelo.tools.velocity_embedding import quiver_autoscale
    from sklearn.neighbors import NearestNeighbors

    X_grid = []
    grs = []
    grid_num = 50 * density
    smooth = 0.5

    for dim_i in range(2):
        m, M = np.min(emb[:, dim_i]), np.max(emb[:, dim_i])
        # m = m - .025 * np.abs(M - m)
        # M = M + .025 * np.abs(M - m)
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(grid_num))
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    n_neighbors = int(emb.shape[0] / 50)
    # nn = NearestNeighbors(n_neighbors=30, n_jobs=-1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(emb)
    dists, neighs = nn.kneighbors(X_grid)
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    if len(velocity_emb.shape) == 2:
        V_grid = (velocity_emb[:, :2][neighs] * weight[:, :, None]).sum(1) / np.maximum(
            1, p_mass
        )[:, None]
    else:
        V_grid = (velocity_emb[:, :2][neighs] * weight[:, :, None, None]).sum(
            1
        ) / np.maximum(1, p_mass)[:, None, None]

    print("V_grid.........")
    p_mass_min *= np.percentile(p_mass, 99) / 100
    if autoscale:
        V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    if uncertain is None:
        return X_grid[p_mass > p_mass_min], V_grid[p_mass > p_mass_min]
    else:
        uncertain = (uncertain[neighs] * weight).sum(1) / np.maximum(1, p_mass)
        return (
            X_grid[p_mass > p_mass_min],
            V_grid[p_mass > p_mass_min],
            uncertain[p_mass > p_mass_min],
        )


def plot_arrow_examples(
    adata,
    v_maps,
    embeds_radian,
    ax=None,
    fig=None,
    cbar=True,
    basis="umap",
    n_sample=30,
    scale=0.0021,
    alpha=0.02,
    index=19,
    scale2=0.04,
    num_certain=3,
    num_total=4,
    p_mass_min=1.0,
    density=0.3,
    arrow_size=4,
):
    from scvelo.plotting.velocity_embedding_grid import default_arrow

    X_grid, V_grid, uncertain = project_grid_points(
        adata.obsm[f"X_{basis}"],
        v_maps,
        get_posterior_sample_angle_uncertainty(embeds_radian / np.pi * 180),
        p_mass_min=p_mass_min,
        density=density,
    )
    norm = Normalize()
    norm.autoscale(uncertain)
    colormap = cm.winter

    indexes = np.argsort(uncertain)[::-1][index : (index + num_total - num_certain)]
    hl, hw, hal = default_arrow(arrow_size)
    quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
    quiver_kwargs.update({"width": 0.001, "headlength": hl / 2})
    quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2, "alpha": 0.6})
    quiver_kwargs.update({"linewidth": 1, "zorder": 3})
    ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=1,
        linewidth=0,
        color="gray",
        alpha=alpha,
    )
    for i in range(n_sample):
        for j in indexes:
            ax.quiver(
                X_grid[j, 0],
                X_grid[j, 1],
                V_grid[j][0][i],
                V_grid[j][1][i],
                ec=colormap(norm(uncertain))[j],
                norm=Normalize(vmin=0, vmax=360),
                scale=scale,
                color=colormap(norm(uncertain))[j],
                **quiver_kwargs,
            )
    indexes = np.argsort(uncertain)[index : (index + num_certain)]
    for i in range(n_sample):
        for j in indexes:
            ax.quiver(
                X_grid[j, 0],
                X_grid[j, 1],
                V_grid[j][0][i],
                V_grid[j][1][i],
                ec=colormap(norm(uncertain))[j],
                scale=scale2,
                color=colormap(norm(uncertain))[j],
                norm=Normalize(vmin=0, vmax=360),
                **quiver_kwargs,
            )
    ax.axis("off")


def set_colorbar(
    smp,
    ax,
    orientation="vertical",
    labelsize=None,
    fig=None,
    position="right",
    rainbow=False,
):
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if position == "right" and (not rainbow):
        cax = inset_axes(ax, width="2%", height="30%", loc=4, borderpad=0)
        cb = fig.colorbar(smp, orientation=orientation, cax=cax)
    else:
        # cax = inset_axes(ax, width="20%", height="90%", loc=4, borderpad=0)
        # cb = fig.colorbar(smp, orientation=orientation, cax=cax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size="8%", pad=0.08)
        cb = fig.colorbar(smp, cax=cax, orientation=orientation, shrink=0.4)

    cb.ax.tick_params(labelsize=labelsize)
    cb.set_alpha(1)
    cb.draw_all()
    cb.locator = MaxNLocator(nbins=2, integer=True)

    if position == "left":
        cb.ax.yaxis.set_ticks_position("left")
    cb.update_ticks()


def us_rainbowplot(
    genes: pd.Index,
    adata: AnnData,
    pos: Dict[str, ndarray],
    data: List[str] = ["st", "ut"],
    cell_state: str = "clusters",
) -> Figure:
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(len(genes), 2)
    fig.set_size_inches(7, 14)
    n = 0
    pos_s = pos[data[0]].mean(0).squeeze()
    pos_u = pos[data[1]].mean(0).squeeze()

    for gene in genes:
        (index,) = np.where(adata.var_names == gene)
        ax1 = ax[n, 1]
        if n == 0:
            ax1.set_title("Rainbow plot")
        ress = pd.DataFrame(
            {
                "cell_time": pos["cell_time"].mean(0).squeeze(),
                "cell_type": adata.obs[cell_state].values,
                "spliced": pos_s[:, index].squeeze(),
                "unspliced": pos_u[:, index].squeeze(),
            }
        )
        if n == 2:
            sns.scatterplot(
                x="cell_time",
                y="spliced",
                data=ress,
                alpha=0.4,
                linewidth=0,
                edgecolor="none",
                hue="cell_type",
                ax=ax1,
                marker="o",
                legend="brief",
                palette="bright",
                s=10,
            )
        else:
            sns.scatterplot(
                x="cell_time",
                y="spliced",
                data=ress,
                alpha=0.4,
                linewidth=0,
                edgecolor="none",
                palette="bright",
                hue="cell_type",
                ax=ax1,
                marker="o",
                legend="brief",
                s=10,
            )

        # ax_twin = ax1.twinx()
        # sns.scatterplot(x='cell_time', y='unspliced', data=ress, alpha=0.4, s=25,
        #                edgecolor="none", palette='bright', hue='cell_type', ax=ax_twin, legend=False, marker="*")
        # ax_twin.set_ylabel("")
        # ax1.set_ylabel("")
        # ax1.set_xlabel("")
        # ax1.tick_params(labelbottom=False)
        # ax_twin.tick_params(labelbottom=False)
        # ax_twin.set_xlabel("")
        ax2 = ax[n, 0]
        ##divider = make_axes_locatable(ax2)
        ##cax = divider.append_axes('right', size='5%', pad=0.05)
        # ax2.set_ylabel("")

        ##ress = pd.DataFrame({"cell_time": pos['cell_time'].mean(0).flatten(),
        ##                     "cell_type": adata.obs[cell_state].values,
        ##                     "unspliced": pos_u[:, index].flatten(),
        ##                     "spliced": pos_s[:, index].flatten()})
        ##im = ax2.scatter(pos_s[:, index], pos_u[:, index], c=ress.cell_time, s=25, cmap = 'seismic')
        ##fig.colorbar(im, cax=cax, orientation='vertical')
        ax2.set_title(gene)
        ax2.set_ylabel("")
        ax2.set_xlabel("")
        sns.scatterplot(
            x="spliced",
            y="unspliced",
            data=ress,
            alpha=0.4,
            s=25,
            edgecolor="none",
            hue="cell_type",
            ax=ax2,
            legend=False,
            marker="*",
            palette="bright",
        )
        if n == 3:
            blue_star = mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=5,
                label="Spliced",
            )
            red_square = mlines.Line2D(
                [],
                [],
                color="black",
                marker="+",
                linestyle="None",
                markersize=5,
                label="Unspliced",
            )
            ax1.legend(handles=[blue_star, red_square], bbox_to_anchor=[2, -0.03])
            ax1.set_xlabel("")
        n += 1
        ax1.legend(bbox_to_anchor=[2, 0.1])
        ax1.tick_params(labelbottom=True)
        ax2.set_xlabel("spliced")
        print(gene)
        ax2.set_title(gene)

        # ax_twin.tick_params(labelbottom=False)
    plt.subplots_adjust(hspace=0.8, wspace=0.6, left=0.1, right=0.91)
    return fig


def rainbowplot(
    volcano_data,
    adata,
    pos,
    fig=None,
    genes=None,
    data=["st", "ut"],
    cell_state="clusters",
    basis="umap",
    num_genes=5,
    add_line=True,
    negative=False,
    scvelo_colors=False,
):
    import matplotlib.pyplot as plt

    if genes is None:
        # selected_genes = volcano_data.sort_values('rank_product', ascending=True).head(20)
        # genes = np.hstack([selected_genes.loc[selected_genes.loc[:, 'time_correlation'] > 0, :].index[:2],
        #           selected_genes.loc[selected_genes.loc[:, 'time_correlation'] < 0, :].index[:2]])

        genes = (
            volcano_data.sort_values("mean_mae", ascending=False)
            .head(300)
            .sort_values("time_correlation", ascending=negative)
            .head(num_genes)
            .index
        )

    import matplotlib

    matplotlib.rcParams.update({"font.size": 7})
    adata.layers["pyro_spliced"] = pos[data[0]].mean(0)

    # fig, ax = plt.subplots(len(genes), 3)
    if fig is None:
        fig = plt.figure(figsize=(5.5, 4.5))

    if scvelo_colors:
        scv.pl.scatter(
            adata,
            basis=basis,
            fontsize=7,
            legend_loc="on data",
            legend_fontsize=7,
            color=cell_state,
            show=False,
        )
        colors = dict(
            zip(adata.obs.state_info.cat.categories, adata.uns["state_info_colors"])
        )
    else:
        clusters = adata.obs.loc[:, cell_state]
        colors = dict(
            zip(
                clusters.cat.categories,
                sns.color_palette("deep", clusters.cat.categories.shape[0]),
            )
        )

    subfigs = fig.subfigures(1, 2, wspace=0.0, width_ratios=[3, 1.5])

    ax = subfigs[0].subplots(len(genes), 2)
    ax_fig2 = subfigs[1].subplots(len(genes), 1)

    n = 0
    st = pos[data[0]].mean(0)
    ut = pos[data[1]].mean(0)
    for gene in genes:
        print(gene)
        (index,) = np.where(adata.var_names == gene)
        ax1 = ax[n, 1]
        ax2 = ax[n, 0]
        ax3 = ax_fig2[n]
        if n == 0:
            ax1.set_title("Rainbow plot", fontsize=7)
            ax2.set_title("Phase portrait", fontsize=7)
        pos_mean_time = pos["cell_time"].mean(0).flatten()
        print(pos_mean_time)
        ress = pd.DataFrame(
            {
                "cell_time": pos_mean_time / pos_mean_time.max(),
                "cell_type": adata.obs[cell_state].values,
                "spliced": st[:, index].flatten(),
                "unspliced": ut[:, index].flatten(),
            }
        )
        # sns.scatterplot(x='cell_time', y='spliced', data=ress, alpha=0.1,
        #                 linewidth=0, edgecolor="none",hue='cell_type',
        #                 #palette=dict(zip(adata.obs.clusters.cat.categories, adata.uns['clusters_colors'])),
        #                 ax=ax1, marker='o', legend=False, s=5)
        sns.scatterplot(
            x="cell_time",
            y="spliced",
            data=ress,
            alpha=0.1,
            linewidth=0,
            edgecolor="none",
            hue="cell_type",
            palette=colors,
            ax=ax1,
            marker="o",
            legend=False,
            s=5,
        )
        # for celltype in scvelo_color:
        #    print(celltype)
        #    ax1.vlines(x=ress.cell_time[ress.cell_type==celltype], ymin=0, ymax=ress.spliced[ress.cell_type==celltype],
        #               colors=scvelo_color[celltype], alpha=0.1)
        if add_line:
            ress = ress.sort_values("cell_time")
            for row in range(ress.shape[0]):
                # ax1.vlines(x=ress.cell_time[row], ymin=0, ymax=ress.spliced[row], colors=scvelo_color[ress.cell_type[row]], alpha=0.1)
                ax1.vlines(
                    x=ress.cell_time[row],
                    ymin=0,
                    ymax=ress.spliced[row],
                    colors=colors[ress.cell_type[row]],
                    alpha=0.1,
                )

        if n == len(genes) - 1:
            ax1.set_xlabel("Shared time", fontsize=7)
        else:
            ax1.set_xlabel("")
        ax1.set_ylabel("")
        t = [0, round(ress["cell_time"].max(), 5)]
        t_label = ["0", "%.1E" % ress["cell_time"].max()]
        ax1.set_xticks(t, t_label, fontsize=7)
        t = [0, round(ress["spliced"].max(), 5)]
        t_label = ["0", "%.1E" % ress["spliced"].max()]
        ax1.set_yticks(t, t_label, fontsize=7)
        # ax1.ticklabel_format(style='sci', axis='both')

        #         t = [ress['unspliced'].min(), ress['unspliced'].max()]
        #         ax1.set_yticks(t,t)
        #         ax_twin = ax1.twinx()
        #         sns.scatterplot(x='cell_time', y='unspliced', data=ress, alpha=0.4, s=25,
        #                         edgecolor="none", hue='cell_type', ax=ax_twin, legend=False, marker="*")
        #         ax_twin.set_ylabel("")
        #         ax1.set_ylabel("")
        #         ax1.set_xlabel("")
        #         ax1.tick_params(labelbottom=False)
        #         ax_twin.tick_params(labelbottom=False)
        #         ax_twin.set_xlabel("")
        #         ax2 = ax[n, 0]
        #         divider = make_axes_locatable(ax2)
        #         cax = divider.append_axes('right', size='5%', pad=0.05)
        #         ax2.set_ylabel("")

        # ress = pd.DataFrame({"cell_time": pos['cell_time'].mean(0).flatten(),
        ress = pd.DataFrame(
            {
                "cell_type": adata.obs[cell_state].values,
                "unspliced": ut[:, index].flatten(),
                "spliced": st[:, index].flatten(),
            }
        )
        # im = ax2.scatter(st[:, index], ut[:, index], c=ress.cell_time, s=25, cmap = 'seismic')
        sns.scatterplot(
            x="spliced",
            y="unspliced",
            data=ress,
            alpha=0.4,
            linewidth=0,
            edgecolor="none",
            hue="cell_type",
            # palette=dict(zip(adata.obs.clusters.cat.categories, adata.uns['clusters_colors'])),
            palette=colors,
            ax=ax2,
            marker="o",
            legend=False,
            s=3,
        )
        ax2.set_xlabel("")

        #         fig.colorbar(im, cax=cax, orientation='vertical')
        ax2.set_ylabel(gene, fontsize=7, rotation=0, labelpad=23)
        if n == len(genes) - 1:
            ax2.set_xlabel("spliced", fontsize=7)
        t = [0, round(ress["unspliced"].max(), 5)]
        t_label = ["0", "%.1E" % ress["unspliced"].max()]
        ax2.set_yticks(t, t_label, fontsize=7)
        t = [0, round(ress["spliced"].max(), 5)]
        t_label = ["0", "%.1E" % ress["spliced"].max()]
        # ax2.ticklabel_format(style='sci', axis='both')
        ax2.set_xticks(t, t_label, fontsize=7)
        # if n == 3:
        #    divider = make_axes_locatable(ax2)
        #    cax = divider.append_axes('bottom', size='5%', pad=0.05)
        #    fig.colorbar(im, cax=cax, orientation='horizontal')
        #         if n == 3:
        #             blue_star = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
        #                                       markersize=5, label='Spliced')
        #             red_square = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
        #                                       markersize=5, label='Unspliced')
        #             ax1.legend(handles=[blue_star, red_square], bbox_to_anchor=[2, -0.03])
        #             ax1.set_xlabel("")
        #         if n == 9:
        #             ax1.tick_params(labelbottom=True)
        #             ax_twin.tick_params(labelbottom=False)
        #             ax2.set_xlabel("spliced")
        # scv.pl.umap(adata, color=gene, ax=ax3, show=False, fontsize=9, title='') #, legend_fontsize=7)
        im = ax3.scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=3,
            c=st[:, index].flatten(),
            cmap="RdBu_r",
        )
        set_colorbar(im, ax3, labelsize=5, fig=subfigs[1], rainbow=True)
        ax3.axis("off")
        if n == 0:
            ax3.set_title("Denoised spliced", fontsize=7)
        n += 1
    sns.despine()
    subfigs[0].subplots_adjust(
        hspace=0.8, wspace=1.4, left=0.26, right=0.94, top=0.92, bottom=0.12
    )
    subfigs[1].subplots_adjust(
        hspace=0.8, wspace=0.4, left=0.2, right=0.7, top=0.92, bottom=0.08
    )
    # subfigs[0].text(0.073, 0.58, "unspliced expression", size=7, rotation="vertical", va="center")
    subfigs[0].text(
        -0.01, 0.58, "unspliced expression", size=7, rotation="vertical", va="center"
    )
    subfigs[0].text(0, 0.97, "f", fontsize=7, fontweight="bold", va="top", ha="right")
    subfigs[0].text(
        0.552, 0.58, "spliced expression", size=7, rotation="vertical", va="center"
    )
    return fig


def plot_state_uncertainty(
    pos, adata, kde=True, data="denoised", top_percentile=0.9, ax=None, basis="umap"
):
    if data == "denoised":
        adata.obs["state_uncertain"] = np.sqrt(
            (
                (pos["st"] - pos["st"].mean(0)) ** 2
                + (pos["ut"] - pos["ut"].mean(0)) ** 2
            ).sum(-1)
        ).mean(0)
    else:
        adata.obs["state_uncertain"] = np.sqrt(
            (
                (pos["s"] - pos["s"].mean(0)) ** 2 + (pos["u"] - pos["u"].mean(0)) ** 2
            ).sum(-1)
        ).mean(0)

    ax = scv.pl.scatter(
        adata,
        basis=basis,
        color="state_uncertain",
        cmap="RdBu_r",
        ax=ax,
        show=False,
        colorbar=True,
        fontsize=7,
    )

    if kde:
        select = adata.obs["state_uncertain"] > np.quantile(
            adata.obs["state_uncertain"], top_percentile
        )
        sns.kdeplot(
            adata.obsm[f"X_{basis}"][:, 0][select],
            adata.obsm[f"X_{basis}"][:, 1][select],
            ax=ax,
            levels=3,
            fill=False,
        )
    else:
        select = None
    return select, ax


import anndata
from scipy.sparse import issparse


def get_clone_trajectory(
    adata, average_start_point=True, global_traj=True, times=[2, 4, 6], clone_num=None
):
    if not average_start_point:
        adata.obsm["clone_vector_emb"] = np.zeros((adata.shape[0], 2))

    adatas = []
    clones = []
    centroids = []
    cen_clones = []
    print(adata.shape)
    adata.obs["clones"] = 0
    if "noWell" in adata.obs.columns:
        for w in adata.obs.Well.unique():
            adata_w = adata[adata.obs.Well == w]
            clone_adata_w = clone_adata[clone_adata.obs.Well == w]
            for j in range(clone_adata_w.shape[1]):
                adata_w.obs["clonei"] = 0
                # belongs to same clone
                adata_w.obs.loc[
                    clone_adata_w[:, j].X.toarray()[:, 0] >= 1, "clonei"
                ] = 1

                if not average_start_point:
                    for i in np.where(
                        (adata_w.obs.time == 2) & (adata_w.obs.clonei == 1)
                    )[0]:
                        next_time = np.where(
                            (adata_w.obs.time == 4) & (adata_w.obs.clonei == 1)
                        )[0]
                        adata_w.obsm["velocity_umap"][i] = (
                            adata_w.obsm["X_umap"][next_time].mean(axis=0)
                            - adata_w.obsm["X_umap"][i]
                        )
                    for i in np.where(
                        (adata_w.obs.time == 4) & (adata_w.obs.clonei == 1)
                    )[0]:
                        next_time = np.where(
                            (adata_w.obs.time == 6) & (adata_w.obs.clonei == 1)
                        )[0]
                        adata_w.obsm["velocity_umap"][i] = (
                            adata_w.obsm["X_umap"][next_time].mean(axis=0)
                            - adata_w.obsm["X_umap"][i]
                        )
                else:
                    time2 = np.where(
                        (adata_w.obs.time == 2) & (adata_w.obs.clonei == 1)
                    )[0]
                    time4 = np.where(
                        (adata_w.obs.time == 4) & (adata_w.obs.clonei == 1)
                    )[0]
                    time6 = np.where(
                        (adata_w.obs.time == 6) & (adata_w.obs.clonei == 1)
                    )[0]
                    if (
                        time2.shape[0] == 0
                        and time4.shape[0] == 0
                        and time6.shape[0] == 0
                    ):
                        continue
                    if (
                        time2.shape[0] > 0
                        and time4.shape[0] == 0
                        and time6.shape[0] > 0
                    ):
                        continue
                    adata_new = anndata.AnnData(
                        np.vstack(
                            [
                                adata_w[time2].X.toarray().mean(axis=0),
                                adata_w[time4].X.toarray().mean(axis=0),
                                adata_w[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        layers={
                            "spliced": np.vstack(
                                [
                                    adata_w[time2]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time4]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time6]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                            "unspliced": np.vstack(
                                [
                                    adata_w[time2]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time4]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time6]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                        },
                        var=adata_w.var,
                    )

                    adata_new.obs.loc[:, "time"] = [2, 4, 6]
                    adata_new.obs.loc[:, "Cell type annotation"] = "Centroid"
                    print(adata_w[time6].obs.clonetype.unique())
                    print(adata_w[time6].obs)

                    adata_new.obs.loc[:, "clonetype"] = adata_w[
                        time6
                    ].obs.clonetype.unique()  # use cell fate from last time point
                    adata_new.obs.loc[:, "clones"] = int(j)
                    if "Well" in adata_w[time6].obs.columns:
                        adata_new.obs.loc[:, "Well"] = adata_w[time6].obs.Well.unique()

                    adata_new.obsm["X_umap"] = np.vstack(
                        [
                            adata_w[time2].obsm["X_umap"].mean(axis=0),
                            adata_w[time4].obsm["X_umap"].mean(axis=0),
                            adata_w[time6].obsm["X_umap"].mean(axis=0),
                        ]
                    )
                    adata_new.obsm["velocity_umap"] = np.vstack(
                        [
                            adata_w.obsm["X_umap"][time4].mean(axis=0)
                            - adata_w.obsm["X_umap"][time2].mean(axis=0),
                            adata_w.obsm["X_umap"][time6].mean(axis=0)
                            - adata_w.obsm["X_umap"][time4].mean(axis=0),
                            np.zeros(2),
                        ]
                    )
                    centroids.append(adata_new)
                    clone_new = anndata.AnnData(
                        np.vstack(
                            [
                                clone_adata_w[time2].X.toarray().mean(axis=0),
                                clone_adata_w[time4].X.toarray().mean(axis=0),
                                clone_adata_w[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        obs=adata_new.obs,
                    )
                    clone_new.var_names = clone_adata.var_names
                    clone_new.var = clone_adata.var
                    # print(clone_new.shape)
                    cen_clones.append(clone_new)

            adata_new = adata_w.concatenate(
                centroids[0].concatenate(centroids[1:]), join="outer"
            )
            clone_new = clone_adata_w.concatenate(
                cen_clones[0].concatenate(cen_clones[1:]), join="outer"
            )
            adatas.append(adata_new)
            clones.append(clone_new)
        return adatas[0].concatenate(adatas[1]), clones[0].concatenate(clones[1])
    else:
        if clone_num is None:
            clone_num = adata.obsm["X_clone"].shape[1]
        for j in range(clone_num):
            print(j)
            adata.obs["clonei"] = 0
            # print('----------aa------')
            if issparse(adata.obsm["X_clone"]):
                adata.obs.loc[adata.obsm["X_clone"].toarray()[:, j] >= 1, "clonei"] = 1
            else:
                adata.obs.loc[adata.obsm["X_clone"][:, j] >= 1, "clonei"] = 1
            # print('----------bb------')

            if not average_start_point:
                for i in np.where((adata.obs.time == 2) & (adata.obs.clonei == 1))[0]:
                    next_time = np.where(
                        (adata.obs.time == 4) & (adata.obs.clonei == 1)
                    )[0]
                    adata.obsm["velocity_umap"][i] = (
                        adata.obsm["X_umap"][next_time].mean(axis=0)
                        - adata.obsm["X_umap"][i]
                    )
                for i in np.where((adata.obs.time == 4) & (adata.obs.clonei == 1))[0]:
                    next_time = np.where(
                        (adata.obs.time == 6) & (adata.obs.clonei == 1)
                    )[0]
                    adata.obsm["velocity_umap"][i] = (
                        adata.obsm["X_umap"][next_time].mean(axis=0)
                        - adata.obsm["X_umap"][i]
                    )
            else:
                if global_traj:
                    times_index = []
                    for t in times:
                        times_index.append(
                            np.where(
                                (adata.obs.time_info == t) & (adata.obs.clonei == 1)
                            )[0]
                        )

                    consecutive_flag = np.array(
                        [int(time.shape[0] > 0) for time in times_index]
                    )
                    consecutive = np.diff(consecutive_flag)
                    if np.sum(consecutive_flag == 1) >= 2 and np.any(
                        consecutive == 0
                    ):  # Must be consecutive time points
                        # print('centroid:', consecutive, times_index)
                        adata_new = anndata.AnnData(
                            np.vstack(
                                [
                                    np.array(adata[time].X.mean(axis=0)).squeeze()
                                    for time in times_index
                                    if time.shape[0] > 0
                                ]
                            ),
                            #                                                     layers={'spliced':
                            #                                                             np.vstack([np.array(adata[time].layers['spliced'].mean(axis=0)) for time in times_index if time.shape[0] > 0]),
                            #                                                             'unspliced':
                            #                                                             np.vstack([np.array(adata[time].layers['unspliced'].mean(axis=0)) for time in times_index if time.shape[0] > 0])
                            #                                                            },
                            var=adata.var,
                        )
                        # print('----------cc------')
                        adata.obs.iloc[
                            np.hstack(
                                [time for time in times_index if time.shape[0] > 0]
                            ),
                            adata.obs.columns.get_loc("clones"),
                        ] = int(j)
                        adata_new.obs.loc[:, "time"] = [
                            t
                            for t, time in zip([2, 4, 6], times_index)
                            if time.shape[0] > 0
                        ]
                        adata_new.obs.loc[:, "clones"] = int(j)
                        adata_new.obs.loc[:, "state_info"] = "Centroid"
                        adata_new.obsm["X_emb"] = np.vstack(
                            [
                                adata[time].obsm["X_emb"].mean(axis=0)
                                for time in times_index
                                if time.shape[0] > 0
                            ]
                        )
                        # print('----------dd------')

                        # print(adata_new.shape)
                        # print(adata_new.obsm['X_umap'])
                        adata_new.obsm["clone_vector_emb"] = np.vstack(
                            [
                                adata_new.obsm["X_emb"][i + 1]
                                - adata_new.obsm["X_emb"][i]
                                for i in range(adata_new.obsm["X_emb"].shape[0] - 1)
                            ]
                            + [np.zeros(2)]
                        )
                        # print('----------ee------')
                        # print(adata_new.obsm['clone_vector_emb'])
                    else:
                        # print('pass-------')
                        continue

                else:
                    time2 = np.where((adata.obs.time == t) & (adata.obs.clonei == 1))[0]
                    time4 = np.where((adata.obs.time == 4) & (adata.obs.clonei == 1))[0]
                    time6 = np.where((adata.obs.time == 6) & (adata.obs.clonei == 1))[0]
                    adata_new = anndata.AnnData(
                        np.vstack(
                            [
                                adata[time2].X.toarray().mean(axis=0),
                                adata[time4].X.toarray().mean(axis=0),
                                adata[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        layers={
                            "spliced": np.vstack(
                                [
                                    adata[time2]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time4]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time6]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                            "unspliced": np.vstack(
                                [
                                    adata[time2]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time4]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time6]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                        },
                        var=adata.var,
                    )

                    print(adata_new.X.sum(axis=1))
                    adata_new.obs.loc[:, "time"] = [2, 4, 6]
                    adata_new.obs.loc[:, "Cell type annotation"] = "Centroid"
                    if not global_traj:
                        adata_new.obs.loc[:, "clonetype"] = adata[
                            time6
                        ].obs.clonetype.unique()  # use cell fate from last time point
                    adata_new.obs.loc[:, "clones"] = j

                    if "noWell" in adata[time6].obs.columns:
                        adata_new.obs.loc[:, "Well"] = adata[time6].obs.Well.unique()

                    adata_new.obsm["X_umap"] = np.vstack(
                        [
                            adata[time2].obsm["X_umap"].mean(axis=0),
                            adata[time4].obsm["X_umap"].mean(axis=0),
                            adata[time6].obsm["X_umap"].mean(axis=0),
                        ]
                    )
                    adata_new.obsm["velocity_umap"] = np.vstack(
                        [
                            adata.obsm["X_umap"][time4].mean(axis=0)
                            - adata.obsm["X_umap"][time2].mean(axis=0),
                            adata.obsm["X_umap"][time6].mean(axis=0)
                            - adata.obsm["X_umap"][time4].mean(axis=0),
                            np.zeros(2),
                        ]
                    )

                    # print(adata_new.obsm['velocity_umap'])
                    clone_new = anndata.AnnData(
                        np.vstack(
                            [
                                clone_adata[time2].X.toarray().mean(axis=0),
                                clone_adata[time4].X.toarray().mean(axis=0),
                                clone_adata[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        obs=adata_new.obs,
                    )
                    clone_new.var_names = clone_adata.var_names
                    clone_new.var = clone_adata.var
                    cen_clones.append(clone_new)
                centroids.append(adata_new)
        print(adata.shape)
        print(len(centroids))
        adata_new = adata.concatenate(
            centroids[0].concatenate(centroids[1:]), join="outer"
        )
        return adata_new


def align_trajectory_diff(
    adatas,
    velocity_embeds,
    density=0.3,
    smooth=0.5,
    input_grid=None,
    input_scale=None,
    min_mass=1.0,
    embed="umap",
    autoscale=False,
    length_cutoff=10,
):
    from scipy.stats import norm as normal
    from scvelo.tools.velocity_embedding import quiver_autoscale
    from sklearn.neighbors import NearestNeighbors

    if input_grid is None and input_scale is None:
        grs = []
        # align embedding points into shared grid across adata
        X_emb = np.vstack([a.obsm[f"X_{embed}"] for a in adatas])
        for dim_i in range(2):
            m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
            m = m - 0.01 * np.abs(M - m)
            M = M + 0.01 * np.abs(M - m)
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
        X_grid = np.vstack([i.flat for i in meshes_tuple]).T
    else:
        scale = input_scale
        X_grid = input_grid

    n_neighbors = int(max([a.shape[0] for a in adatas]) / 50)

    results = [X_grid]
    p_mass_list = []
    for adata, velocity_embed in zip(adatas, velocity_embeds):
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(adata.obsm[f"X_{embed}"])
        dists, neighs = nn.kneighbors(X_grid)
        weight = normal.pdf(x=dists, scale=scale)
        # how many cells around a grid points
        p_mass = weight.sum(1)
        V_grid = (velocity_embed[neighs] * weight[:, :, None]).sum(1) / np.maximum(
            1, p_mass
        )[:, None]
        if autoscale:
            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)
        results.append(V_grid)
        p_mass_list.append(p_mass)

    from functools import reduce

    if input_grid is None and input_scale is None:
        min_mass *= np.percentile(np.hstack(p_mass_list), 99) / 100
        mass_index = reduce(
            np.intersect1d, [np.where(p_mass > min_mass)[0] for p_mass in p_mass_list]
        )

    results = np.hstack(results)
    results = results[mass_index]
    length_filter = np.sqrt((results[:, 2:4] ** 2).sum(1)) > length_cutoff
    return results[length_filter]
