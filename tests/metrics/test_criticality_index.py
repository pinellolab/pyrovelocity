import pytest
import scvelo as scv
from pyrovelocity.metrics.criticality_index import calculate_criticality_index

@pytest.fixture(scope="module")
def adata_fixture():
    return scv.datasets.simulation(
        random_seed=0,
        n_obs=10,
        n_vars=3,
        alpha=5,
        beta=0.5,
        gamma=0.3,
        alpha_=0,
        switches=[1, 5, 10],
        noise_model="gillespie",
    )

def test_calculate_criticality_index(adata_fixture):
    criticality_index, _, _, _ = calculate_criticality_index(adata_fixture)
    assert isinstance(criticality_index, float)
    assert 0 <= criticality_index <= 1

