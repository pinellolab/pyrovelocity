from typing import List

import pandas as pd
import pytest

from pyrovelocity.analysis.analyze import (
    pareto_frontier_genes,
)


@pytest.fixture
def sample_volcano_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean_mae": [0.1, 0.2, 0.3, 0.4, 0.5],
            "time_correlation": [0.5, 0.6, 0.7, 0.8, 0.9],
        },
        index=["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"],
    )


@pytest.fixture
def sample_volcano_data_with_ribosomal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean_mae": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "time_correlation": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        index=["Gene1", "Rpl1", "Gene3", "Rps2", "Gene5", "Gene6"],
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
        sample_volcano_data, num_genes=5, max_iters=1000
    )
    assert result == ["Gene5", "Gene4", "Gene3", "Gene2", "Gene1"]


def test_pareto_frontier_genes_ribosomal_filtering(
    sample_volcano_data_with_ribosomal
):
    result = pareto_frontier_genes(
        sample_volcano_data_with_ribosomal, num_genes=4, max_iters=1000
    )
    assert "Rpl1" not in result
    assert "Rps2" not in result
    assert len(result) == 4


def test_pareto_frontier_genes_max_iters(sample_volcano_data):
    result = pareto_frontier_genes(
        sample_volcano_data, num_genes=5, max_iters=1
    )
    assert len(result) < 5


def test_pareto_frontier_genes_fewer_genes(sample_volcano_data):
    result = pareto_frontier_genes(
        sample_volcano_data, num_genes=10, max_iters=1000
    )
    assert len(result) == 5
