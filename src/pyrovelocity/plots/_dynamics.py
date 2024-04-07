import matplotlib.pyplot as plt

__all__ = [
    "plot_deterministic_simulation_phase_portrait",
    "plot_deterministic_simulation_trajectories",
]


def plot_deterministic_simulation_trajectories(solution, title_prefix=""):
    ts = solution.ts
    unspliced, spliced = solution.ys[:, 0], solution.ys[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # linear scale
    axes[0].plot(
        ts,
        unspliced,
        "o-",
        label=r"$u^{\ast}$",
        color="blue",
        markersize=5,
        linewidth=1.5,
        alpha=0.5,
    )
    axes[0].plot(
        ts,
        spliced,
        "o-",
        label=r"$s^{\ast}$",
        color="green",
        markersize=5,
        linewidth=1.5,
        alpha=0.5,
    )
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Concentration")
    axes[0].set_title(f"{title_prefix} Temporal Trajectories (Linear Scale)")
    axes[0].legend()

    # log scale
    axes[1].plot(
        ts,
        unspliced,
        "o-",
        label=r"$u^{\ast}$",
        color="blue",
        markersize=5,
        linewidth=1.5,
        alpha=0.5,
    )
    axes[1].plot(
        ts,
        spliced,
        "o-",
        label=r"$s^{\ast}$",
        color="green",
        markersize=5,
        linewidth=1.5,
        alpha=0.5,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Concentration")
    axes[1].set_title(f"{title_prefix} Temporal Trajectories (Log Scale)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_deterministic_simulation_phase_portrait(solution, title_prefix=""):
    unspliced, spliced = solution.ys[:, 0], solution.ys[:, 1]
    norm_time = solution.ts

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # linear scale
    sc = axes[0].scatter(
        spliced,
        unspliced,
        c=norm_time,
        cmap="viridis",
        label="Phase Portrait",
    )
    axes[0].set_xlabel(r"$s^{\ast}$")
    axes[0].set_ylabel(r"$u^{\ast}$")
    axes[0].set_title(f"{title_prefix} Phase Portrait (Linear Scale)")
    plt.colorbar(sc, ax=axes[0], label="Time")

    # log scale
    axes[1].scatter(
        spliced,
        unspliced,
        c=norm_time,
        cmap="viridis",
        label="Phase Portrait",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$s^{\ast}$")
    axes[1].set_ylabel(r"$u^{\ast}$")
    axes[1].set_title(f"{title_prefix} Phase Portrait (Log Scale)")
    plt.colorbar(sc, ax=axes[1], label="Time")
    plt.tight_layout()
    plt.show()
