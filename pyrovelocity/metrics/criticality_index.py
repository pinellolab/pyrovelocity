import anndata
from typing import Tuple


def calculate_criticality_index(
    adata: anndata.AnnData,
) -> Tuple[float, float, float, float]:
    """
    Calculates a criticality index for multimodal single-cell RNA-seq
    data stored in an AnnData object.

    Parameters:
        adata (anndata.AnnData): The AnnData object containing single-cell
                                 RNA-seq data with unspliced and spliced layers.

    Returns:
        criticality_index (float): The criticality index.
        pcc_mean (float): Mean of the Pearson correlation coefficient.
        sd_mean (float): Mean of the standard deviation.
        H_mean (float): Mean of the conditional entropy.

    Examples:
        >>> import scvelo as scv
        >>> adata = scv.datasets.simulation(
        ...     random_seed=0,
        ...     n_obs=10,
        ...     n_vars=3,
        ...     alpha=5,
        ...     beta=0.5,
        ...     gamma=0.3,
        ...     alpha_=0,
        ...     switches=[1, 5, 10],
        ...     noise_model="gillespie",
        ... )
        >>> print(f"unspliced\\n{adata.layers['unspliced'].T}")
        >>> print(f"spliced\\n{adata.layers['spliced'].T}")
        >>> (
        ...     criticality_index,
        ...     pcc_mean,
        ...     sd_mean,
        ...     H_mean
        ... ) = calculate_criticality_index(adata)
        >>> print(f"PCC Mean: {pcc_mean}")
        >>> print(f"SD Mean: {sd_mean}")
        >>> print(f"H Mean: {H_mean}")
        >>> print(f"Criticality Index: {criticality_index:.4f}")
    """
