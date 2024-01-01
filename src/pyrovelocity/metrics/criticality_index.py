from typing import Tuple

import anndata
import numpy as np
import pandas as pd


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

    if adata.X.size == 0:
        raise ValueError("AnnData object must not be empty.")

    if "unspliced" not in adata.layers or "spliced" not in adata.layers:
        raise KeyError(
            "AnnData object must contain both 'unspliced' and 'spliced' layers."
        )

    unspliced_data = adata.layers["unspliced"].T
    spliced_data = adata.layers["spliced"].T

    pcc_unspliced = np.corrcoef(unspliced_data)
    pcc_spliced = np.corrcoef(spliced_data)

    pcc_mean = np.mean(
        (
            pcc_unspliced[np.triu_indices(unspliced_data.shape[0], k=1)]
            + pcc_spliced[np.triu_indices(spliced_data.shape[0], k=1)]
        )
        / 2
    )

    sd_mean = np.mean(
        np.hstack(
            (np.std(unspliced_data, axis=1), np.std(spliced_data, axis=1))
        )
    )

    def conditional_entropy(data):
        df = pd.DataFrame(data)
        joint_prob = (
            df.apply(lambda x: x.value_counts() / len(x))
            .fillna(0)
            .values.flatten()
        )
        joint_prob = joint_prob.reshape(data.shape[1], -1)
        marginal_prob = np.sum(joint_prob, axis=1)
        cond_prob = joint_prob / marginal_prob[:, None]
        cond_entropy = -np.nansum(joint_prob * np.log2(cond_prob))
        return cond_entropy

    H_unspliced = conditional_entropy(unspliced_data)
    H_spliced = conditional_entropy(spliced_data)

    H_mean = (H_unspliced + H_spliced) / 2

    criticality_index = (pcc_mean * sd_mean) / H_mean
    return criticality_index, pcc_mean, sd_mean, H_mean
