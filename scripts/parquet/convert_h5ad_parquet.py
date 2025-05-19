from pathlib import Path

from anndata import AnnData, read_h5ad
from beartype import beartype

from pqdata import read_anndata, write_anndata
from pyrovelocity.utils import print_anndata


@beartype
def convert_h5ad_to_pqdata(
    h5ad_file: Path,
    pq_path: Path,
) -> tuple[AnnData, AnnData]:
    """
    Converts an AnnData file to pqdata format.

    Args:
        h5ad_file (Path): Path to the AnnData file.
        pq_path (Path): Path to the pqdata file.
    """
    adata = read_h5ad(h5ad_file)
    print_anndata(adata)
    write_anndata(
        data=adata,
        path=pq_path,
        compression="zstd",
        overwrite=True,
    )
    adata_pq = read_anndata(path=pq_path)
    print_anndata(adata_pq)
    return (adata, adata_pq)


if __name__ == "__main__":
    file_path = Path("data")
    data_set_name = "postprocessed_pancreas_50_7"

    h5ad_file = file_path / data_set_name / f"{data_set_name}.h5ad"
    pq_path = file_path / data_set_name / f"{data_set_name}.pqdata"

    adata, adata_pq = convert_h5ad_to_pqdata(h5ad_file, pq_path)
