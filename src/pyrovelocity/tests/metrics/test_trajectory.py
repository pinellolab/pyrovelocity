"""Tests for trajectory evaluation metrics."""

import numpy as np

from pyrovelocity.metrics.trajectory import (
    cross_boundary_correctness,
    inner_cluster_coherence,
)


def test_cross_boundary_correctness_pancreas(adata_postprocessed_pancreas_50_7):
    """Test cross-boundary correctness calculation for pancreas dataset."""
    transitions = [
        ("Ngn3 high EP", "Pre-endocrine"),
        ("Pre-endocrine", "Alpha"),
        ("Pre-endocrine", "Beta"),
        ("Pre-endocrine", "Delta"),
        ("Pre-endocrine", "Epsilon"),
    ]

    scores, mean_score = cross_boundary_correctness(
        adata_postprocessed_pancreas_50_7,
        k_cluster="clusters",
        cluster_edges=transitions,
        k_velocity="velocity_pyro",
        x_emb="X_umap",
    )

    assert isinstance(scores, dict)
    assert isinstance(mean_score, float)
    assert all(isinstance(score, float) for score in scores.values())
    assert all(-1 <= score <= 1 for score in scores.values())
    assert -1 <= mean_score <= 1

    assert all((u, v) in scores for u, v in transitions)


def test_inner_cluster_coherence_pancreas(adata_postprocessed_pancreas_50_7):
    """Test inner-cluster coherence calculation for pancreas dataset."""
    scores, mean_score = inner_cluster_coherence(
        adata_postprocessed_pancreas_50_7,
        k_cluster="clusters",
        k_velocity="velocity_pyro",
    )

    assert isinstance(scores, dict)
    assert isinstance(mean_score, float)
    assert all(isinstance(score, float) for score in scores.values())
    assert all(0 <= score <= 1 for score in scores.values())
    assert 0 <= mean_score <= 1

    clusters = np.unique(adata_postprocessed_pancreas_50_7.obs["clusters"])
    assert all(cluster in scores for cluster in clusters)
