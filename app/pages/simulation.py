import scvelo as scv
import streamlit as st
from google.cloud import storage


adata = scv.datasets.simulation(
    random_seed=0,
    n_obs=100,
    n_vars=12,
    alpha=5,
    beta=0.5,
    gamma=0.3,
    alpha_=0,
    noise_model="normal",  # vs "gillespie" broken in 0.2.4
)

scv.set_figure_params(vector_friendly=False, transparent=False, facecolor="white")


def st_show():
    # Plot initial scatter plots
    axs = scv.pl.scatter(
        adata,
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        ncols=4,
        nrows=3,
        xlim=[-1, 20],
        ylim=[-1, 20],
        show=False,
        dpi=300,
        figsize=(7, 5),
    )
    st.pyplot(axs[0].get_figure(), format="png", dpi=300)

    # Recover dynamics and plot
    scv.tl.recover_dynamics(adata)

    axs2 = scv.pl.scatter(
        adata,
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        ncols=4,
        nrows=3,
        xlim=[-1, 20],
        ylim=[-1, 20],
        color=["true_t"],
        show=False,
        dpi=300,
        figsize=(7, 5),
    )
    st.pyplot(axs2[0].get_figure(), format="png", dpi=300)

    # c = (
    #     alt.Chart(iris)
    #     .mark_point()
    #     .encode(x="petalLength", y="petalWidth", color="species")
    # )
    # st.altair_chart(c, use_container_width=True)
