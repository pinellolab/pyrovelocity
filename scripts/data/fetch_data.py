import pyroe
import scanpy as sc


def download_pyroe_data(dataset_number, save_path):
    pq_dict = pyroe.load_processed_quant([dataset_number])
    pq_ds20 = pq_dict[dataset_number]

    adata = pq_ds20.anndata
    adata.layers["spliced"] = adata.X

    print(adata)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(adata)

    adata.write(save_path)


if __name__ == "__main__":
    download_pyroe_data(20, "pbmc5k_unlabeled.h5ad")
    # download_pyroe_data(9, "pbmc20k_unlabeled.h5ad")
