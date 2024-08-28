from pathlib import Path

import daft
import matplotlib.pyplot as plt

__all__ = [
    "variable_initial_condition_model_plate_diagram",
    "variable_initial_condition_multiple_timepoints_model_plate_diagram",
]


def variable_initial_condition_model_plate_diagram(
    output_path: str
    | Path = "variable_initial_condition_model_plate_diagram.pdf",
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["text.usetex"] = True

    pgm = daft.PGM(line_width=1.2)

    optional_color = "#ff6a14"
    optional_color_params = {"ec": optional_color}

    # hyperparameters
    pgm.add_node(
        "mu_init",
        r"$\mu_{0}$",
        0.5,
        6,
        fixed=True,
        plot_params=optional_color_params | {"fc": optional_color},
    )
    pgm.add_node(
        "sigma_init",
        r"$\sigma_{0}^2$",
        1.5,
        6,
        fixed=True,
        plot_params=optional_color_params | {"fc": optional_color},
    )
    pgm.add_node("mu_theta", r"$\mu_{\theta}$", 2.5, 6, fixed=True)
    pgm.add_node("sigma_theta", r"$\sigma_{\theta}^2$", 3.5, 6, fixed=True)
    pgm.add_node("mu_sigma", r"$\mu_{\sigma}$", 4.5, 6, fixed=True)
    pgm.add_node("sigma_sigma", r"$\sigma_{\sigma}^2$", 5.5, 6, fixed=True)

    # latent variables for gene-specific parameters
    pgm.add_node(
        "us_0i",
        r"$(u,s)_{0i}$",
        1,
        5,
        fontsize=7,
        scale=1.0,
        plot_params=optional_color_params,
    )
    pgm.add_node("t_0i", r"$t_{0i}$", 2, 5, plot_params=optional_color_params)
    pgm.add_node("theta_i", r"$\theta_i$", 3, 5)
    pgm.add_node("sigma_ui", r"$\sigma_{ui}$", 4, 5)
    pgm.add_node("sigma_si", r"$\sigma_{si}$", 5, 5)

    # latent variables for cell-specific outcomes
    pgm.add_node(
        "u_ij",
        r"$u_{ij}$",
        2,
        4,
        scale=1.0,
        shape="rectangle",
    )
    pgm.add_node(
        "s_ij",
        r"$s_{ij}$",
        4,
        4,
        scale=1.0,
        shape="rectangle",
    )

    # observed data
    pgm.add_node("t_j", r"$t_j$", 6.0, 3.25)
    pgm.add_node(
        "u_obs_ij",
        r"$\hat{u}_{ij}$",
        2,
        2.5,
        scale=1.0,
        observed=True,
    )
    pgm.add_node(
        "s_obs_ij",
        r"$\hat{s}_{ij}$",
        4,
        2.5,
        scale=1.0,
        observed=True,
    )

    # edges
    edge_params = {"head_length": 0.3, "head_width": 0.25, "lw": 0.7}
    optional_color_params.update({"fc": optional_color})
    optional_color_params.update(edge_params)
    pgm.add_edge("mu_init", "us_0i", plot_params=optional_color_params)
    pgm.add_edge("sigma_init", "us_0i", plot_params=optional_color_params)
    pgm.add_edge("mu_init", "t_0i", plot_params=optional_color_params)
    pgm.add_edge("sigma_init", "t_0i", plot_params=optional_color_params)
    pgm.add_edge("mu_theta", "theta_i", plot_params=edge_params)
    pgm.add_edge("sigma_theta", "theta_i", plot_params=edge_params)
    pgm.add_edge("mu_sigma", "sigma_ui", plot_params=edge_params)
    pgm.add_edge("sigma_sigma", "sigma_ui", plot_params=edge_params)
    pgm.add_edge("mu_sigma", "sigma_si", plot_params=edge_params)
    pgm.add_edge("sigma_sigma", "sigma_si", plot_params=edge_params)

    pgm.add_edge("us_0i", "u_ij", plot_params=optional_color_params)
    pgm.add_edge("t_0i", "u_ij", plot_params=optional_color_params)
    pgm.add_edge("us_0i", "s_ij", plot_params=optional_color_params)
    pgm.add_edge("t_0i", "s_ij", plot_params=optional_color_params)
    pgm.add_edge("theta_i", "s_ij", plot_params=edge_params)
    pgm.add_edge("theta_i", "u_ij", plot_params=edge_params)

    pgm.add_edge("u_ij", "u_obs_ij", plot_params=edge_params)
    pgm.add_edge("s_ij", "s_obs_ij", plot_params=edge_params)
    pgm.add_edge("sigma_ui", "u_obs_ij", plot_params=edge_params)
    pgm.add_edge("sigma_si", "s_obs_ij", plot_params=edge_params)

    pgm.add_edge("t_j", "u_ij", plot_params=edge_params)
    pgm.add_edge("t_j", "s_ij", plot_params=edge_params)

    # plates
    pgm.add_plate(
        [0.5, 1.2, 5, 4.4],
        label=r"Genes $i \in \{1, \ldots, G\}$",
        shift=-0.1,
        fontsize=12,
    )
    pgm.add_plate(
        [1.0, 1.8, 5.5, 2.75],
        label=r"Cells $j \in \{1, \ldots, N\}$",
        shift=-0.1,
        fontsize=12,
    )

    pgm.render()

    for ext in ["", ".png"]:
        pgm.savefig(
            f"{output_path}{ext}",
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
            transparent=False,
        )


def variable_initial_condition_multiple_timepoints_model_plate_diagram(
    output_path: str
    | Path = "variable_initial_condition_multiple_timepoints_model_plate_diagram.pdf",
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["text.usetex"] = True

    pgm = daft.PGM(line_width=1.2)

    # hyperparameters
    pgm.add_node("mu_init", r"$\mu_{0}$", 0.5, 6, fixed=True)
    pgm.add_node("sigma_init", r"$\sigma_{0}^2$", 1.5, 6, fixed=True)
    pgm.add_node("mu_theta", r"$\mu_{\theta}$", 2.5, 6, fixed=True)
    pgm.add_node("sigma_theta", r"$\sigma_{\theta}^2$", 3.5, 6, fixed=True)
    pgm.add_node("mu_sigma", r"$\mu_{\sigma}$", 4.5, 6, fixed=True)
    pgm.add_node("sigma_sigma", r"$\sigma_{\sigma}^2$", 5.5, 6, fixed=True)

    # latent variables for gene-specific parameters
    # pgm.add_node("us_0i", r"$\mathbf{x}_{0i}$", 1, 5, scale=1.0)
    pgm.add_node("us_0i", r"$(u,s)_{0i}$", 1, 5, fontsize=7, scale=1.0)
    pgm.add_node("t_0i", r"$t_{0i}$", 2, 5)
    pgm.add_node("theta_i", r"$\theta_i$", 3, 5)
    pgm.add_node("sigma_ui", r"$\sigma_{ui}$", 4, 5)
    pgm.add_node("sigma_si", r"$\sigma_{si}$", 5, 5)

    # latent variables for cell-specific outcomes
    pgm.add_node(
        "u_ij",
        r"${u}^k_{ij}$",
        2,
        3.8,
        scale=1.0,
        shape="rectangle",
    )
    pgm.add_node(
        "s_ij",
        r"${s}^k_{ij}$",
        4,
        3.8,
        scale=1.0,
        shape="rectangle",
    )

    # observed data
    pgm.add_node("t_j", r"${t}^k_j$", 5.9, 3.1)
    pgm.add_node(
        "u_obs_ij",
        r"$\hat{u}{}^{k}_{ij}$",
        2,
        2.4,
        scale=1.0,
        observed=True,
    )
    pgm.add_node(
        "s_obs_ij",
        r"$\hat{s}{}^{k}_{ij}$",
        4,
        2.4,
        scale=1.0,
        observed=True,
    )

    # edges
    edge_params = {"head_length": 0.3, "head_width": 0.25, "lw": 0.7}
    pgm.add_edge("mu_init", "us_0i", plot_params=edge_params)
    pgm.add_edge("sigma_init", "us_0i", plot_params=edge_params)
    pgm.add_edge("mu_init", "t_0i", plot_params=edge_params)
    pgm.add_edge("sigma_init", "t_0i", plot_params=edge_params)
    pgm.add_edge("mu_theta", "theta_i", plot_params=edge_params)
    pgm.add_edge("sigma_theta", "theta_i", plot_params=edge_params)
    pgm.add_edge("mu_sigma", "sigma_ui", plot_params=edge_params)
    pgm.add_edge("sigma_sigma", "sigma_ui", plot_params=edge_params)
    pgm.add_edge("mu_sigma", "sigma_si", plot_params=edge_params)
    pgm.add_edge("sigma_sigma", "sigma_si", plot_params=edge_params)

    pgm.add_edge("us_0i", "u_ij", plot_params=edge_params)
    pgm.add_edge("t_0i", "u_ij", plot_params=edge_params)
    pgm.add_edge("us_0i", "s_ij", plot_params=edge_params)
    pgm.add_edge("t_0i", "s_ij", plot_params=edge_params)
    pgm.add_edge("theta_i", "s_ij", plot_params=edge_params)
    pgm.add_edge("theta_i", "u_ij", plot_params=edge_params)

    pgm.add_edge("u_ij", "u_obs_ij", plot_params=edge_params)
    pgm.add_edge("s_ij", "s_obs_ij", plot_params=edge_params)
    pgm.add_edge("sigma_ui", "u_obs_ij", plot_params=edge_params)
    pgm.add_edge("sigma_si", "s_obs_ij", plot_params=edge_params)

    pgm.add_edge("t_j", "u_ij", plot_params=edge_params)
    pgm.add_edge("t_j", "s_ij", plot_params=edge_params)

    # plates
    pgm.add_plate(
        [0.4, 1.0, 5, 4.5],
        label=r"$i \in \{1, \ldots, G\}$",
        shift=-0.1,
        fontsize=12,
    )
    pgm.add_plate(
        [0.8, 1.4, 5.9, 3.2],
        label=r"$j \in \{1, \ldots, N\}$",
        shift=-0.1,
        fontsize=12,
    )
    pgm.add_plate(
        [1.2, 1.8, 5.2, 2.5],
        label=r"$k \in \{1, \ldots, K_j\}$",
        shift=-0.1,
        fontsize=12,
    )

    pgm.render()

    for ext in ["", ".png"]:
        pgm.savefig(
            f"{output_path}{ext}",
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
            transparent=False,
        )
