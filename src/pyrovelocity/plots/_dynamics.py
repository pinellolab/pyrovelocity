import shutil

import jax.numpy as jnp
import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import List
from diffrax import Solution
from matplotlib import colors

__all__ = [
    "plot_deterministic_simulation_phase_portrait",
    "plot_deterministic_simulation_trajectories",
]


@beartype
def plot_deterministic_simulation_trajectories(
    solution: Solution,
    state_labels: List[str] = [r"$u^{\ast}$", r"$s^{\ast}$"],
    xlabel: str = r"$t^{\ast}$",
    ylabel: str = r"fraction of $u^{\ast}_{ss}$",
    title_prefix: str = "",
    colormap_name: str = "cividis",
):
    times = solution.ts
    logarithmizable_times = jnp.where(times <= 1e-4, 1e-2, times)
    unspliced, spliced = solution.ys[:, 0], solution.ys[:, 1]
    logarithmizable_unspliced = jnp.where(
        jnp.abs(unspliced) <= 3e-8, 1e-8, jnp.abs(unspliced)
    )
    logarithmizable_spliced = jnp.where(
        jnp.abs(spliced) <= 3e-8, 1e-8, jnp.abs(spliced)
    )

    with plt.style.context(["pyrovelocity.styles.common"]):
        if not shutil.which("latex"):
            plt.rc(
                "text",
                usetex=False,
            )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cmap = plt.get_cmap(colormap_name)
        color_brown = cmap(0.05)
        color_blue_green = cmap(0.95)

        # linear scale
        axes[0].plot(
            times,
            unspliced,
            "o-",
            label=state_labels[0],
            color=color_brown,
            markersize=5,
            linewidth=1.5,
            alpha=0.5,
        )
        axes[0].plot(
            times,
            spliced,
            "o-",
            label=state_labels[1],
            color=color_blue_green,
            markersize=5,
            linewidth=1.5,
            alpha=0.5,
        )
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(
            f"{title_prefix} Temporal Trajectories (Linear Scale)"
        )
        axes[0].legend()

        # log scale
        axes[1].plot(
            logarithmizable_times,
            logarithmizable_unspliced,
            "o-",
            label=state_labels[0],
            color=color_brown,
            markersize=5,
            linewidth=1.5,
            alpha=0.5,
        )
        axes[1].plot(
            logarithmizable_times,
            logarithmizable_spliced,
            "o-",
            label=state_labels[1],
            color=color_blue_green,
            markersize=5,
            linewidth=1.5,
            alpha=0.5,
        )
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        axes[1].set_title(f"{title_prefix} Temporal Trajectories (Log Scale)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


@beartype
def plot_deterministic_simulation_phase_portrait(
    solution: Solution,
    xlabel: str = r"$s^{\ast}$",
    ylabel: str = r"$u^{\ast}$",
    zlabel: str = r"$t^{\ast}$",
    title_prefix: str = "",
    colormap_name: str = "cividis",
):
    times = solution.ts
    logarithmizable_times = jnp.where(times <= 1e-4, 1e-2, times)

    unspliced, spliced = solution.ys[:, 0], solution.ys[:, 1]

    with plt.style.context(["pyrovelocity.styles.common"]):
        if not shutil.which("latex"):
            plt.rc(
                "text",
                usetex=False,
            )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # linear scale
        sc_linear = axes[0].scatter(
            spliced,
            unspliced,
            c=times,
            cmap=colormap_name,
            label="Phase Portrait",
        )
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(f"{title_prefix} Phase Portrait (Linear Scale)")
        plt.colorbar(sc_linear, ax=axes[0], label=zlabel)

        # log scale
        sc_log = axes[1].scatter(
            spliced,
            unspliced,
            c=logarithmizable_times,
            cmap=colormap_name,
            norm=colors.LogNorm(),
            label="Phase Portrait",
        )
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        axes[1].set_title(f"{title_prefix} Phase Portrait (Log Scale)")
        plt.colorbar(sc_log, ax=axes[1], label=zlabel)
        plt.tight_layout()
        plt.show()
