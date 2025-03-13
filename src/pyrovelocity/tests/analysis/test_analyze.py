from typing import List

import numpy as np
import pandas as pd
import pytest

from pyrovelocity.analysis.analyze import (
    mae_per_gene,
    pareto_frontier_genes,
    top_mae_genes,
)


@pytest.fixture
def sample_volcano_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean_mae": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "time_correlation": [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.0],
        },
        index=[f"Gene{i}" for i in range(1, 9)],
    )


@pytest.fixture
def sample_volcano_data_with_ribosomal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean_mae": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "time_correlation": [
                -0.9,
                -0.6,
                -0.3,
                0,
                0.3,
                0.6,
                0.9,
                1.0,
                -0.5,
                0.5,
            ],
        },
        index=[
            "Gene1",
            "Rpl1",
            "Gene3",
            "Rps2",
            "Gene5",
            "Gene6",
            "Gene7",
            "Gene8",
            "Gene9",
            "Gene10",
        ],
    )


@pytest.fixture
def small_volcano_data() -> pd.DataFrame:
    """A small dataset with fewer than 8*min_genes_per_bin genes."""
    return pd.DataFrame(
        {
            "mean_mae": [0.1, 0.2, 0.3, 0.4, 0.5],
            "time_correlation": [-0.9, -0.6, -0.3, 0, 0.3],
        },
        index=[f"Gene{i}" for i in range(1, 6)],
    )


@pytest.mark.parametrize("xp", [np])
def test_mae_per_gene_basic(xp):
    true_counts = xp.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    )
    pred_counts = xp.array(
        [
            [1.1, 2.2, 3.3],
            [1.1, 2.2, 3.3],
            [1.1, 2.2, 3.3],
            [1.1, 2.2, 3.3],
        ]
    )
    mae = mae_per_gene(pred_counts, true_counts)
    assert xp.allclose(mae, xp.array([-0.1, -0.1, -0.1]), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("xp", [np])
def test_mae_per_gene_zero_counts(xp):
    true_counts = xp.array(
        [
            [10, 15, 0],
            [20, 25, 0],
        ]
    )
    pred_counts = xp.array(
        [
            [12, 14, 0],
            [18, 26, 0],
        ]
    )
    mae = mae_per_gene(pred_counts, true_counts)
    assert xp.allclose(
        mae, xp.array([-0.133, -0.05, -0.0]), rtol=1e-2, atol=1e-2
    )


def test_pareto_frontier_genes_basic(sample_volcano_data):
    result = pareto_frontier_genes(
        sample_volcano_data, num_genes=3, max_iters=1000
    )
    assert isinstance(result, List)
    assert len(result) == 3
    assert all(gene in sample_volcano_data.index for gene in result)


def test_pareto_frontier_genes_order(sample_volcano_data):
    result = pareto_frontier_genes(
        sample_volcano_data, num_genes=8, max_iters=1000
    )
    time_correlations = [
        sample_volcano_data.loc[gene, "time_correlation"] for gene in result
    ]
    assert time_correlations == sorted(time_correlations, reverse=True)
    assert len(result) == 8


def test_pareto_frontier_genes_ribosomal_filtering(
    sample_volcano_data_with_ribosomal
):
    result = pareto_frontier_genes(
        sample_volcano_data_with_ribosomal, num_genes=6, max_iters=1000
    )
    assert "Rpl1" not in result
    assert "Rps2" not in result
    assert len(result) == 6


def test_pareto_frontier_genes_max_iters(sample_volcano_data):
    result = pareto_frontier_genes(
        sample_volcano_data, num_genes=8, max_iters=1
    )
    assert len(result) < 8


def test_pareto_frontier_genes_fewer_genes(sample_volcano_data):
    result = pareto_frontier_genes(
        sample_volcano_data, num_genes=10, max_iters=1000
    )
    assert len(result) == 8


def test_top_mae_genes_basic(sample_volcano_data):
    result = top_mae_genes(
        sample_volcano_data, mae_top_percentile=50, min_genes_per_bin=1
    )
    assert isinstance(result, List)
    assert 4 <= len(result) <= 8
    assert all(gene in sample_volcano_data.index for gene in result)


def test_top_mae_genes_order(sample_volcano_data):
    result = top_mae_genes(
        sample_volcano_data, mae_top_percentile=100, min_genes_per_bin=1
    )
    assert result == [
        "Gene8",
        "Gene7",
        "Gene6",
        "Gene5",
        "Gene4",
        "Gene3",
        "Gene2",
        "Gene1",
    ]


def test_top_mae_genes_ribosomal_filtering(sample_volcano_data_with_ribosomal):
    result = top_mae_genes(
        sample_volcano_data_with_ribosomal,
        mae_top_percentile=50,
        min_genes_per_bin=1,
    )
    assert "Rpl1" not in result
    assert "Rps2" not in result
    assert 4 <= len(result) <= 8


def test_top_mae_genes_min_genes_per_bin(sample_volcano_data):
    result = top_mae_genes(
        sample_volcano_data,
        mae_top_percentile=10,
        min_genes_per_bin=2,
    )
    assert len(result) >= 8


def test_top_mae_genes_time_correlation_sorting(sample_volcano_data):
    result = top_mae_genes(
        sample_volcano_data, mae_top_percentile=50, min_genes_per_bin=1
    )
    time_correlations = [
        sample_volcano_data.loc[gene, "time_correlation"] for gene in result
    ]
    assert time_correlations == sorted(time_correlations, reverse=True)


def test_top_mae_genes_percentile_threshold(sample_volcano_data):
    result_10 = top_mae_genes(
        sample_volcano_data, mae_top_percentile=10, min_genes_per_bin=1
    )
    result_50 = top_mae_genes(
        sample_volcano_data, mae_top_percentile=50, min_genes_per_bin=1
    )
    assert len(result_10) <= len(result_50)


def test_top_mae_genes_small_dataset(small_volcano_data):
    """Test that top_mae_genes returns all genes for a small dataset."""
    result = top_mae_genes(
        small_volcano_data, mae_top_percentile=10, min_genes_per_bin=1
    )
    assert len(result) == len(small_volcano_data)
    assert set(result) == set(small_volcano_data.index)


def test_top_mae_genes_percentile_capping():
    """Test that mae_top_percentile is capped at 100 if it exceeds that value."""
    df = pd.DataFrame(
        {
            "mean_mae": [0.1, 0.2, 0.3],
            "time_correlation": [-0.9, 0, 0.9],
        },
        index=["Gene1", "Gene2", "Gene3"],
    )
    # This would have raised an error before, but now should be capped at 100
    result = top_mae_genes(df, mae_top_percentile=200, min_genes_per_bin=1)
    # With percentile=100, all genes should be included
    assert len(result) == len(df)


def test_top_mae_genes_invalid_percentile():
    """Test that an error is still raised for invalid percentiles <= 0."""
    with pytest.raises(ValueError):
        top_mae_genes(pd.DataFrame(), mae_top_percentile=0, min_genes_per_bin=1)
    with pytest.raises(ValueError):
        top_mae_genes(
            pd.DataFrame(), mae_top_percentile=-10, min_genes_per_bin=1
        )


def test_top_mae_genes_invalid_min_genes():
    """Test that an error is raised for invalid min_genes_per_bin values."""
    # Use a non-empty DataFrame to avoid the empty DataFrame check
    df = pd.DataFrame(
        {
            "mean_mae": [0.1, 0.2, 0.3],
            "time_correlation": [-0.9, 0, 0.9],
        },
        index=["Gene1", "Gene2", "Gene3"],
    )
    with pytest.raises(ValueError):
        top_mae_genes(df, mae_top_percentile=50, min_genes_per_bin=-1)
    with pytest.raises(ValueError):
        top_mae_genes(df, mae_top_percentile=50, min_genes_per_bin=0)


def test_top_mae_genes_empty_dataframe():
    """Test that an empty list is returned for an empty DataFrame."""
    result = top_mae_genes(
        pd.DataFrame(), mae_top_percentile=50, min_genes_per_bin=1
    )
    assert isinstance(result, list)
    assert len(result) == 0
