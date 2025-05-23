#!/usr/bin/env python3

import anndata
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import os
import concurrent.futures
from pathlib import Path
from pyrovelocity.utils import print_anndata


def process_simulation(i, path_prefix):
    """Process a single simulation dataset.

    Args:
        i: Simulation index
        path_prefix: Path prefix for the simulation data

    Returns:
        The simulation index if successful, None otherwise
    """
    try:
        print(f"Processing simulation {i}...")

        zarr_path = os.path.join(path_prefix, f"simulation_{i}.zarr")

        if not os.path.exists(zarr_path):
            print(f"Error: Zarr file not found at {zarr_path}")
            return None

        adata = anndata.io.read_zarr(zarr_path)

        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        scv.tl.velocity_graph(adata, vkey="true_velocity", n_jobs=1)

        print_anndata(adata)

        vector_field_basis = "umap"
        cell_state = "leiden"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")
        Path(plots_dir).mkdir(exist_ok=True)
        vector_field_plot = os.path.join(plots_dir, f"simulation_{i}.pdf")

        fig, ax = plt.subplots()
        ax.axis("off")

        scv.pl.velocity_embedding_grid(
            adata,
            basis=vector_field_basis,
            color=cell_state,
            title="",
            vkey="true_velocity",
            s=15,
            alpha=1,
            linewidth=0.5,
            ax=ax,
            show=False,
            legend_loc="right margin",
            density=0.4,
            scale=0.2,
            arrow_size=2,
            arrow_length=2,
            arrow_color="black",
        )
        for ext in ["", ".png"]:
            fig.savefig(
                f"{vector_field_plot}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )
        plt.close(fig)
        print(f"Completed simulation {i}")
        return i
    except Exception as e:
        print(f"Error processing simulation {i}: {str(e)}")
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(
        script_dir,
        os.getenv("DATA_DIR", ""),
    ))

    if not path_prefix.endswith(os.path.sep):
        path_prefix += os.path.sep

    print(f"Using data path: {path_prefix}")

    if not os.path.isdir(path_prefix):
        print(f"Warning: Data directory not found at {path_prefix}")
        print("Please ensure the data directory exists before running this script.")

    max_workers = max(1, os.cpu_count() - 1)
    print(f"Using {max_workers} workers for parallel processing")

    simulation_indices = list(range(1, 50))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_simulation, i, path_prefix): i
            for i in simulation_indices
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result is not None:
                    print(f"Successfully processed simulation {index}")
                else:
                    print(f"Failed to process simulation {index}")
            except Exception as e:
                print(f"Exception occurred while processing simulation {index}: {str(e)}")


if __name__ == "__main__":
    main()
