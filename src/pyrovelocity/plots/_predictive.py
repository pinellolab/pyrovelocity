from os import PathLike
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
from anndata import AnnData
from beartype import beartype
from matplotlib.figure import FigureBase
from pyro.infer import Predictive
from pyro.infer import infer_discrete

from pyrovelocity.logging import configure_logging
from pyrovelocity.models import PyroVelocity


__all__ = ["posterior_curve", "extrapolate_prediction_sample_predictive"]

logger = configure_logging(__name__)


@beartype
def posterior_curve(
    adata: AnnData,
    posterior_samples: Dict[str, np.ndarray],
    gene_set: List[str],
    data_model: str,
    model_path: PathLike | str,
    output_directory: PathLike | str,
) -> List[FigureBase]:
    grid_cell_time = posterior_samples["cell_time"]

    logger.info(
        "Extrapolating prediction samples for predictive posterior plots"
    )

    (
        grid_time_samples_ut,
        grid_time_samples_st,
        grid_time_samples_u0,
        grid_time_samples_s0,
        grid_time_samples_uinf,
        grid_time_samples_sinf,
        grid_time_samples_uscale,
        grid_time_samples_state,
        grid_time_samples_t0,
        grid_time_samples_dt_switching,
    ) = extrapolate_prediction_sample_predictive(
        grid_cell_time,
        model_path,
        adata,
        grid_time_points=500,
    )

    logger.info("Plotting posterior predictive phase portraits")

    output_fig_objects = []
    for figi, gene in enumerate(gene_set):
        (index,) = np.where(adata.var_names == gene)
        # print(adata.shape, index, posterior_samples["st_mean"].shape)

        fig, ax = plt.subplots(3, 4)
        fig.set_size_inches(15, 10)
        ax = ax.flatten()
        for sample in range(4):
            t0_sample = posterior_samples["t0"][sample][:, index[0]].flatten()
            cell_time_sample = posterior_samples["cell_time"][sample].flatten()
            cell_time_sample_max = cell_time_sample.max()
            cell_time_sample_min = cell_time_sample.min()

            colors = np.array(["gray", "blue"])
            mask_t0_sample = (cell_time_sample >= t0_sample).astype(int)
            cell_colors = colors[mask_t0_sample]

            colors = np.array(["gray", "blue", "red"])
            grid_mask_t0_sample = (
                grid_cell_time.mean(0).flatten() >= t0_sample
            ).astype("float32")

            cell_gene_state_grid = grid_time_samples_state[0][
                :, index[0]
            ].astype("float32")

            grid_mask_t0_sample = grid_mask_t0_sample + cell_gene_state_grid
            grid_mask_t0_sample = grid_mask_t0_sample.astype(int)
            grid_mask_t0_sample[
                grid_cell_time.mean(0).flatten() < t0_sample
            ] = 0
            grid_cell_colors = colors[grid_mask_t0_sample]
            # print(grid_time_samples_st.shape)

            im = ax[sample].scatter(
                posterior_samples["st_mean"][:, index[0]],
                posterior_samples["ut_mean"][:, index[0]],
                s=3,
                linewidth=0,
                # color=cell_colors,
                color=grid_cell_colors,
                alpha=0.6,
            )

            im = ax[sample].scatter(
                grid_time_samples_st[sample][:, index[0]],
                grid_time_samples_ut[sample][:, index[0]],
                s=15,
                marker="*",
                linewidth=0,
                alpha=0.2,
                c=grid_cell_colors,
            )
            ax[sample + 4].scatter(
                grid_cell_time.mean(0).flatten(),
                posterior_samples["ut_mean"][:, index[0]],
                s=3,
                linewidth=0,
                marker=".",
                color=grid_cell_colors,
                alpha=0.3,
            )
            ax[sample + 4].scatter(
                grid_cell_time[sample].flatten(),
                grid_time_samples_ut[sample][:, index[0]],
                s=15,
                marker=">",
                linewidth=0,
                color=grid_cell_colors,
                alpha=0.5,
            )
            ax[sample + 4].set_title("Unspliced", fontsize=7)
            ax[sample + 4].set_ylabel("Unspliced (Ut)", fontsize=7)
            ax[sample + 8].scatter(
                grid_cell_time.mean(0).flatten(),
                posterior_samples["st_mean"][:, index[0]],
                s=3,
                marker="*",
                linewidth=0,
                color=grid_cell_colors,
                alpha=0.3,
            )
            ax[sample + 8].scatter(
                grid_cell_time[sample].flatten(),
                grid_time_samples_st[sample][:, index[0]],
                s=15,
                marker="<",
                linewidth=0,
                color=grid_cell_colors,
                alpha=0.3,
            )
            ax[sample + 8].set_title("Spliced", fontsize=7)
            ax[sample + 8].set_ylabel("Spliced (St)", fontsize=7)

            u0 = grid_time_samples_u0[sample][:, index[0]].flatten()
            uscale = grid_time_samples_uscale[sample][:, index[0]].flatten()
            s0 = grid_time_samples_s0[sample][:, index[0]].flatten()
            u_inf = grid_time_samples_uinf[sample][:, index[0]].flatten()
            s_inf = grid_time_samples_sinf[sample][:, index[0]].flatten()

            t0_sample = grid_time_samples_t0[sample][:, index[0]].flatten()
            dt_switching_sample = grid_time_samples_dt_switching[sample][
                :, index[0]
            ].flatten()

            ##u0 = posterior_samples['u_offset'][sample][:, index[0]].flatten()
            ##s0 = posterior_samples['s_offset'][sample][:, index[0]].flatten()
            ##u_inf = posterior_samples['u_inf'][sample][:, index[0]].flatten()
            ##s_inf = posterior_samples['s_inf'][sample][:, index[0]].flatten()
            ##switching = posterior_samples['switching'][sample][:, index[0]].flatten()
            ##dt_switching = posterior_samples['dt_switching'][sample][:, index[0]].flatten()

            ax[sample + 4].scatter(
                t0_sample,
                u0 * uscale,
                s=80,
                marker="p",
                linewidth=0.5,
                c="purple",
                alpha=0.8,
            )
            ax[sample + 4].scatter(
                t0_sample + dt_switching_sample,
                u_inf * uscale,
                s=80,
                marker="*",
                linewidth=0.5,
                c="black",
                alpha=0.8,
            )

            ax[sample + 8].scatter(
                t0_sample,
                s0,
                s=80,
                marker="p",
                linewidth=0.5,
                c="purple",
                alpha=0.8,
            )
            ax[sample + 8].scatter(
                t0_sample + dt_switching_sample,
                s_inf,
                s=80,
                marker="p",
                linewidth=0.5,
                c="black",
                alpha=0.8,
            )

            ax[sample].scatter(
                s0, u0 * uscale, s=60, marker="p", linewidth=0.5, c="purple"
            )
            ax[sample].scatter(
                s_inf,
                u_inf * uscale,
                s=60,
                marker="p",
                linewidth=0.5,
                c="black",
            )
            # ax[sample].plot(grid_time_samples_st[sample][:, index[0]],
            #                grid_time_samples_ut[sample][:, index[0]],
            #                linestyle="--", linewidth=3, color='g')
            # if sample == 0:
            #     print(gene, u0 * uscale, s0)
            #     print(gene, u_inf * uscale, s_inf)
            #     print(
            #         t0_sample,
            #         dt_switching_sample,
            #         cell_time_sample_min,
            #         cell_time_sample_max,
            #         (cell_time_sample <= t0_sample).sum(),
            #     )
            #     print(cell_time_sample.shape)

            switching = t0_sample + dt_switching_sample
            state0 = (cell_gene_state_grid == 0) & (
                cell_time_sample <= switching
            )
            state0_false = (cell_gene_state_grid == 0) & (
                cell_time_sample > switching
            )
            state1 = (cell_gene_state_grid == 1) & (
                cell_time_sample >= switching
            )
            state1_false = (cell_gene_state_grid == 1) & (
                cell_time_sample < switching
            )

            ax[sample].set_title(
                f"{gene} model 2 sample {sample}\nt0>celltime:{(t0_sample>cell_time_sample_max)} {(t0_sample>cell_time_sample).sum()}\nstate0: {state0.sum()} {state0_false.sum()} state1: {state1.sum()} {state1_false.sum()}",
                fontsize=6.5,
            )
            ax[sample].set_xlim(
                0,
                max(
                    [
                        np.max(posterior_samples["st_mean"][:, index[0]]) * 1.1,
                        np.max(grid_time_samples_st[sample][:, index[0]]),
                        s0 * 1.1,
                        s_inf * 1.1,
                    ]
                ),
            )
            ax[sample].set_ylim(
                0,
                max(
                    [
                        np.max(posterior_samples["ut_mean"][:, index[0]]) * 1.1,
                        np.max(grid_time_samples_ut[sample][:, index[0]]),
                        u0 * uscale * 1.1,
                        u_inf * uscale * 1.05,
                    ]
                ),
            )
            fig.colorbar(im, ax=ax[sample])
        fig.tight_layout()
        fig.savefig(
            f"{output_directory}/{data_model}_{gene}.png",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )
        output_fig_objects.append(fig)
        plt.close(fig)

    return output_fig_objects


def extrapolate_prediction_sample_predictive(
    posterior_time,
    data_model_path,
    adata,
    grid_time_points=1000,
    use_gpu="cpu",
):
    PyroVelocity.setup_anndata(adata)
    model = PyroVelocity(
        adata, add_offset=False, guide_type="auto_t0_constraint"
    )
    model = model.load_model(
        dir_path=data_model_path,
        adata=adata,
        use_gpu=use_gpu,
    )
    print(data_model_path)

    scdl = model._make_data_loader(adata=adata, indices=None, batch_size=1000)

    posterior_samples_list = []
    for tensor_dict in scdl:
        # print("--------------------")
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        u_log_library_mean = tensor_dict["u_lib_size_mean"]
        s_log_library_mean = tensor_dict["s_lib_size_mean"]
        u_log_library_scale = tensor_dict["u_lib_size_scale"]
        s_log_library_scale = tensor_dict["s_lib_size_scale"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        # dummy_obs = (
        #     torch.tensor(u_obs).to("cuda:0"),
        #     torch.tensor(s_obs).to("cuda:0"),
        #     torch.tensor(u_log_library).to("cuda:0"),
        #     torch.tensor(s_log_library).to("cuda:0"),
        #     torch.tensor(u_log_library_mean).to("cuda:0"),
        #     torch.tensor(s_log_library_mean).to("cuda:0"),
        #     torch.tensor(u_log_library_scale).to("cuda:0"),
        #     torch.tensor(s_log_library_scale).to("cuda:0"),
        #     torch.tensor(ind_x).to("cuda:0"),
        #     None,
        #     None,
        # )
        dummy_obs = (
            torch.tensor(u_obs),
            torch.tensor(s_obs),
            torch.tensor(u_log_library),
            torch.tensor(s_log_library),
            torch.tensor(u_log_library_mean),
            torch.tensor(s_log_library_mean),
            torch.tensor(u_log_library_scale),
            torch.tensor(s_log_library_scale),
            torch.tensor(ind_x),
            None,
            None,
        )

        posterior_samples = {}
        posterior_samples_batch_sample = []
        for sample in range(5):
            guide_trace = pyro.poutine.trace(model.module.guide).get_trace(
                *dummy_obs
            )
            trained_model = pyro.poutine.replay(
                model.module.model, trace=guide_trace
            )
            model_discrete = infer_discrete(
                trained_model, temperature=0, first_available_dim=-3
            )
            trace = pyro.poutine.trace(model_discrete).get_trace(*dummy_obs)
            map_estimate_cell_gene_state = trace.nodes["cell_gene_state"][
                "value"
            ]
            alpha = trace.nodes["alpha"]["value"]
            beta = trace.nodes["beta"]["value"]
            gamma = trace.nodes["gamma"]["value"]
            t0 = trace.nodes["t0"]["value"]
            dt_switching = trace.nodes["dt_switching"]["value"]
            cell_time = trace.nodes["cell_time"]["value"]

            if "u_offset" in trace.nodes:
                u_offset = trace.nodes["u_offset"]["value"]
                s_offset = trace.nodes["s_offset"]["value"]
                u_scale = trace.nodes["u_scale"]["value"]
            else:
                u_offset = alpha.new_zeros(alpha.shape)
                s_offset = alpha.new_zeros(alpha.shape)
                u_scale = alpha.new_ones(alpha.shape)
            posterior_samples_batch_sample.append(
                {
                    "cell_gene_state": map_estimate_cell_gene_state.unsqueeze(
                        -3
                    ),
                    "alpha": alpha.unsqueeze(-2).unsqueeze(-3),
                    "beta": beta.unsqueeze(-2).unsqueeze(-3),
                    "gamma": gamma.unsqueeze(-2).unsqueeze(-3),
                    "u_offset": u_offset.unsqueeze(-2).unsqueeze(-3),
                    "s_offset": s_offset.unsqueeze(-2).unsqueeze(-3),
                    "u_scale": u_scale.unsqueeze(-2).unsqueeze(-3),
                    "dt_switching": dt_switching.unsqueeze(-2).unsqueeze(-3),
                    "cell_time": cell_time.unsqueeze(-3),
                    "t0": t0.unsqueeze(-2).unsqueeze(-3),
                }
            )

        for key in posterior_samples_batch_sample[0].keys():
            posterior_samples[key] = torch.tensor(
                np.concatenate(
                    [
                        posterior_samples_batch_sample[j][key]
                        .cpu()
                        .detach()
                        .numpy()
                        for j in range(len(posterior_samples_batch_sample))
                    ],
                    axis=-3,
                )
            )
            # ).to("cuda:0")

        posterior_samples_new_tmp = Predictive(
            pyro.poutine.uncondition(
                model.module.model,
            ),
            posterior_samples,
        )(*dummy_obs)
        for key in posterior_samples:
            posterior_samples_new_tmp[key] = posterior_samples[key]
        posterior_samples_list.append(posterior_samples_new_tmp)

    # print(len(posterior_samples_list))
    posterior_samples_new = {}
    for key in posterior_samples_list[0].keys():
        if posterior_samples_list[0][key].shape[-2] == 1:
            posterior_samples_new[key] = posterior_samples_list[0][key]
        else:
            posterior_samples_new[key] = torch.concat(
                [element[key] for element in posterior_samples_list], axis=-2
            )
    # posterior_samples_new = model.generate_posterior_samples(
    #    adata=adata, batch_size=512, num_samples=8
    # )

    # for key in posterior_samples_new.keys():
    #     print(posterior_samples_new[key].shape)

    grid_time_samples_ut = posterior_samples_new["ut"]
    grid_time_samples_st = posterior_samples_new["st"]
    grid_time_samples_uinf = posterior_samples_new["u_inf"]
    grid_time_samples_sinf = posterior_samples_new["s_inf"]
    if "u_offset" in posterior_samples_new:
        grid_time_samples_u0 = posterior_samples_new["u_offset"]
        grid_time_samples_s0 = posterior_samples_new["s_offset"]
    else:
        grid_time_samples_u0 = np.zeros(grid_time_samples_uinf.shape)
        grid_time_samples_s0 = np.zeros(grid_time_samples_sinf.shape)

    grid_time_samples_t0 = posterior_samples_new["t0"]
    grid_time_samples_dt_switching = posterior_samples_new["dt_switching"]
    if "u_offset" in posterior_samples_new:
        grid_time_samples_uscale = posterior_samples_new["u_scale"]
    else:
        grid_time_samples_uscale = np.ones(grid_time_samples_uinf.shape)

    grid_time_samples_state = posterior_samples_new["cell_gene_state"]
    # print(grid_time_samples_state.shape)
    # print(grid_time_samples_uscale.shape)
    # print(grid_time_samples_ut.shape)
    # print(grid_time_samples_st.shape)
    if isinstance(grid_time_samples_state, np.ndarray):
        return (
            grid_time_samples_ut,
            grid_time_samples_st,
            grid_time_samples_u0,
            grid_time_samples_s0,
            grid_time_samples_uinf,
            grid_time_samples_sinf,
            grid_time_samples_uscale,
            grid_time_samples_state,
            grid_time_samples_t0,
            grid_time_samples_dt_switching,
        )
    else:
        return (
            grid_time_samples_ut.cpu().detach().numpy(),
            grid_time_samples_st.cpu().detach().numpy(),
            grid_time_samples_u0.cpu().detach().numpy(),
            grid_time_samples_s0.cpu().detach().numpy(),
            grid_time_samples_uinf.cpu().detach().numpy(),
            grid_time_samples_sinf.cpu().detach().numpy(),
            grid_time_samples_uscale.cpu().detach().numpy(),
            grid_time_samples_state.cpu().detach().numpy(),
            grid_time_samples_t0.cpu().detach().numpy(),
            grid_time_samples_dt_switching.cpu().detach().numpy(),
        )
