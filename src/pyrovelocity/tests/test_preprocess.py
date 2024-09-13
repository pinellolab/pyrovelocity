"""Tests for `pyrovelocity.tasks.preprocess` module."""


def test_load_preprocess():
    from pyrovelocity.tasks import preprocess

    print(preprocess.__file__)
    
def test_compute_metacells():
    from pyrovelocity.tasks.preprocess import compute_metacells
    from pyrovelocity.tests.synthetic_AnnData import synthetic_AnnData
    import scanpy as sc
    adata_rna = synthetic_AnnData(seed = 1)
    adata_atac = synthetic_AnnData(seed = 2)
    sc.tl.pca(adata_rna, n_comps=3)
    compute_metacells(adata_rna, adata_atac,
                 latent_key = 'X_pca',
                  resolution = 1,
                  celltype_key = 'cell_type')
    
