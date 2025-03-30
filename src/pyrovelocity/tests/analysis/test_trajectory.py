import numpy as np
from anndata import AnnData

from pyrovelocity.analysis.trajectory import (
    align_trajectory_diff,
    get_clone_trajectory,
)


def test_get_clone_trajectory_all_clones(adata_larry_multilineage_50_6):
    """Test get_clone_trajectory with all clones (clone_num=None)."""
    adata = adata_larry_multilineage_50_6.copy()

    if "X_clone" not in adata.obsm:
        n_cells = adata.n_obs
        n_clones = 5
        clone_matrix = np.zeros((n_cells, n_clones))
        for i in range(min(n_clones, n_cells // 10)):
            clone_matrix[i * 10 : (i + 1) * 10, i] = 1
        adata.obsm["X_clone"] = clone_matrix

    if "timepoint" not in adata.obs:
        adata.obs["timepoint"] = [2, 4, 6] * (adata.n_obs // 3) + [2, 4, 6][
            : adata.n_obs % 3
        ]

    result = get_clone_trajectory(adata=adata, clone_num=None)

    assert isinstance(result, AnnData)
    assert "clone_vector_emb" in result.obsm

    if "clone_id" in result.obs:
        unique_clones = result.obs["clone_id"].nunique()
        expected_clones = min(5, adata.n_obs // 10)  # Based on our setup
        assert unique_clones <= expected_clones


def test_align_trajectory_diff(adata_larry_multilineage_50_6):
    """Smoke test for align_trajectory_diff function.

    This test verifies that align_trajectory_diff runs without errors and returns
    the expected type of output, using the same pattern as observed in
    plot_lineage_fate_correlation.
    """
    adata = adata_larry_multilineage_50_6.copy()

    if "X_clone" not in adata.obsm:
        n_cells = adata.n_obs
        n_clones = 5
        clone_matrix = np.zeros((n_cells, n_clones))
        for i in range(min(n_clones, n_cells // 10)):
            clone_matrix[i * 10 : (i + 1) * 10, i] = 1
        adata.obsm["X_clone"] = clone_matrix

    if "timepoint" not in adata.obs:
        adata.obs["timepoint"] = [2, 4, 6] * (adata.n_obs // 3) + [2, 4, 6][
            : adata.n_obs % 3
        ]

    if "time_info" not in adata.obs:
        adata.obs["time_info"] = adata.obs.get("timepoint", 0)

    if "X_emb" not in adata.obsm:
        adata.obsm["X_emb"] = np.random.normal(size=(adata.n_obs, 2))

    if "velocity_emb" not in adata.obsm:
        adata.obsm["velocity_emb"] = np.random.normal(size=(adata.n_obs, 2))

    adata_clone = get_clone_trajectory(adata, clone_num=None)

    adata2 = adata.copy()

    density = 0.35
    result = align_trajectory_diff(
        [adata_clone, adata, adata2],
        [
            adata_clone.obsm.get(
                "clone_vector_emb", np.zeros((adata_clone.n_obs, 2))
            ),
            adata.obsm.get("velocity_emb", np.zeros((adata.n_obs, 2))),
            adata2.obsm.get("velocity_emb", np.zeros((adata2.n_obs, 2))),
        ],
        embed="emb",
        density=density,
    )

    assert isinstance(result, np.ndarray)
    # Result should be a 2D array with each row containing:
    # [x, y, vx1, vy1, vx2, vy2, vx3, vy3] where:
    # - (x, y) is the grid point
    # - (vx1, vy1) is the vector from adata_clone
    # - (vx2, vy2) is the vector from adata
    # - (vx3, vy3) is the vector from adata2
    assert result.ndim == 2
    assert result.shape[1] == 8  # Grid points (2) + 3 vector fields (6)
