import os
from pathlib import Path

import duckdb
import ibis

from anndata import AnnData, read_h5ad
from beartype import beartype
from huggingface_hub import HfApi, RepoUrl

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


def read_parquet_files_duckdb(pq_path: Path, data_set_name: str) -> Path:
    """
    Read all parquet files in a directory tree using DuckDB and save to a database file.

    Args:
        pq_path (Path): Path to the directory containing parquet files.
        data_set_name (str): Name of the dataset, used for the database filename.

    This function uses DuckDB to find all parquet files in the directory tree,
    reads each as a separate table, displays their schemas, and saves the database to a file.
    """
    db_path = Path(f"{data_set_name}_duckdb.db")
    print(f"Creating DuckDB database at: {db_path.absolute()}")

    conn = duckdb.connect(str(db_path))

    glob_pattern = str(pq_path / "**" / "*.parquet")

    file_list = conn.execute(f"SELECT * FROM glob('{glob_pattern}')").fetchall()

    if not file_list:
        print(f"No parquet files found in {pq_path}")
        return None

    print(f"Found {len(file_list)} parquet files in {pq_path}")

    tables = {}

    for idx, (file_path,) in enumerate(file_list):
        rel_path = Path(file_path).relative_to(pq_path)
        table_name = str(rel_path).replace("/", "_").replace(".parquet", "")

        print(f"\n[{idx + 1}/{len(file_list)}] Reading: {rel_path}")

        try:
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM read_parquet('{file_path}')
            """)
            tables[table_name] = file_path

            schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()

            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            print(f"Table: {table_name}")
            print(f"Source: {file_path}")
            print(f"Rows: {row_count}")
            print(f"Columns: {len(schema_info)}")

            print(f"\n{'Column Name':<30} {'Type':<20} {'Null':<10}")
            print("-" * 60)
            for column_name, column_type, null, _, _, _ in schema_info:
                null_str = "NULL" if null == "YES" else "NOT NULL"
                print(f"{column_name:<30} {column_type:<20} {null_str:<10}")

            if row_count > 0:
                print("\nSample data (first 3 rows):")
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()
                print(sample)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\nSuccessfully created {len(tables)} tables in the database.")
    print("Table names:")
    for name in sorted(tables.keys()):
        print(f"  - {name}")

    conn.close()

    print(f"\nDatabase saved to: {db_path.absolute()}")
    print(f"Database size: {db_path.stat().st_size / (1024 * 1024):.2f} MB")

    return db_path


def read_parquet_files_duckdb_views(pq_path: Path, data_set_name: str) -> Path:
    """
    Read all parquet files in a directory tree using DuckDB and create views instead of tables.

    Args:
        pq_path (Path): Path to the directory containing parquet files.
        data_set_name (str): Name of the dataset, used for the database filename.

    This function uses DuckDB to find all parquet files in the directory tree,
    creates a view for each file (instead of copying the data), and displays their schemas.
    """
    db_path = Path(f"{data_set_name}_duckdb_views.db")
    print(f"Creating DuckDB views database at: {db_path.absolute()}")

    conn = duckdb.connect(str(db_path))

    glob_pattern = str(pq_path / "**" / "*.parquet")

    file_list = conn.execute(f"SELECT * FROM glob('{glob_pattern}')").fetchall()

    if not file_list:
        print(f"No parquet files found in {pq_path}")
        return None

    print(f"Found {len(file_list)} parquet files in {pq_path}")

    views = {}

    for idx, (file_path,) in enumerate(file_list):
        rel_path = Path(file_path).relative_to(pq_path)
        view_name = str(rel_path).replace("/", "_").replace(".parquet", "")

        print(f"\n[{idx + 1}/{len(file_list)}] Creating view for: {rel_path}")

        try:
            conn.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet('{file_path}')
            """)
            views[view_name] = file_path

            schema_info = conn.execute(f"DESCRIBE {view_name}").fetchall()

            row_count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]

            print(f"View: {view_name}")
            print(f"Source: {file_path}")
            print(f"Rows: {row_count}")
            print(f"Columns: {len(schema_info)}")

            print(f"\n{'Column Name':<30} {'Type':<20} {'Null':<10}")
            print("-" * 60)
            for column_name, column_type, null, _, _, _ in schema_info:
                null_str = "NULL" if null == "YES" else "NOT NULL"
                print(f"{column_name:<30} {column_type:<20} {null_str:<10}")

            if row_count > 0:
                print("\nSample data (first 3 rows):")
                sample = conn.execute(f"SELECT * FROM {view_name} LIMIT 3").fetchdf()
                print(sample)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\nSuccessfully created {len(views)} views in the database.")
    print("View names:")
    for name in sorted(views.keys()):
        print(f"  - {name}")

    conn.close()

    print(f"\nDatabase with views saved to: {db_path.absolute()}")
    print(f"Database size: {db_path.stat().st_size / (1024 * 1024):.2f} MB")

    return db_path


def read_parquet_files_as_tables_ibis(pq_path: Path, data_set_name: str) -> Path:
    """
    Read each parquet file in a directory tree as a separate table using Ibis with DuckDB backend
    and save to a database file.

    Args:
        pq_path (Path): Path to the directory containing parquet files.
        data_set_name (str): Name of the dataset, used for the database filename.

    This function uses Ibis with DuckDB backend to find all parquet files in the directory tree,
    read each as a separate table, display their schemas, and save the database to a file.
    """
    ibis.options.interactive = True

    db_path = Path(f"{data_set_name}_ibis.db")
    print(f"Creating DuckDB database via Ibis at: {db_path.absolute()}")

    conn = ibis.duckdb.connect(str(db_path))

    parquet_files = list(pq_path.glob("**/*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {pq_path}")
        return None

    print(f"Found {len(parquet_files)} parquet files in {pq_path}")

    tables = {}

    for idx, file_path in enumerate(parquet_files):
        rel_path = file_path.relative_to(pq_path)
        table_name = str(rel_path).replace("/", "_").replace(".parquet", "")

        print(f"\n[{idx + 1}/{len(parquet_files)}] Reading: {rel_path}")

        try:
            table = conn.read_parquet(file_path, table_name=table_name)

            conn.create_table(table_name, table, overwrite=True)

            table = conn.table(table_name)
            tables[table_name] = table

            schema = table.schema()
            print(f"Schema for {table_name}:")
            print(schema)

            row_count = table.count().execute()
            col_count = len(table.columns)
            print(f"Dimensions: {row_count} rows × {col_count} columns")

            if row_count > 0:
                print("\nSample data (first 3 rows):")
                print(table.limit(3))

        except Exception as e:
            print(f"Error reading {rel_path}: {e}")

    print(f"\nSuccessfully loaded {len(tables)} tables into the database.")
    print("Table names:")
    for name in sorted(tables.keys()):
        print(f"  - {name}")

    conn.con.close()

    print(f"\nDatabase saved to: {db_path.absolute()}")
    print(f"Database size: {db_path.stat().st_size / (1024 * 1024):.2f} MB")

    return db_path


def read_parquet_files_as_views_ibis(pq_path: Path, data_set_name: str) -> Path:
    """
    Read each parquet file in a directory tree as a view using Ibis with DuckDB backend
    and save to a database file.

    Args:
        pq_path (Path): Path to the directory containing parquet files.
        data_set_name (str): Name of the dataset, used for the database filename.

    This function uses Ibis with DuckDB backend to find all parquet files in the directory tree,
    create a view for each file (instead of copying the data), and display their schemas.
    """
    ibis.options.interactive = True

    db_path = Path(f"{data_set_name}_ibis_views.db")
    print(f"Creating DuckDB views database via Ibis at: {db_path.absolute()}")

    conn = ibis.duckdb.connect(str(db_path))

    parquet_files = list(pq_path.glob("**/*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {pq_path}")
        return None

    print(f"Found {len(parquet_files)} parquet files in {pq_path}")

    views = {}

    for idx, file_path in enumerate(parquet_files):
        rel_path = file_path.relative_to(pq_path)
        view_name = str(rel_path).replace("/", "_").replace(".parquet", "")

        print(f"\n[{idx + 1}/{len(parquet_files)}] Creating view for: {rel_path}")

        try:
            conn.raw_sql(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet('{file_path}')
            """)

            view = conn.table(view_name)
            views[view_name] = view

            schema = view.schema()
            print(f"Schema for {view_name}:")
            print(schema)

            row_count = view.count().execute()
            col_count = len(view.columns)
            print(f"Dimensions: {row_count} rows × {col_count} columns")

            if row_count > 0:
                print("\nSample data (first 3 rows):")
                print(view.limit(3))

        except Exception as e:
            print(f"Error reading {rel_path}: {e}")

    print(f"\nSuccessfully created {len(views)} views in the database.")
    print("View names:")
    for name in sorted(views.keys()):
        print(f"  - {name}")

    conn.con.close()

    print(f"\nDatabase with views saved to: {db_path.absolute()}")
    print(f"Database size: {db_path.stat().st_size / (1024 * 1024):.2f} MB")

    return db_path


def analyze_parquet_database_ibis(pq_path: Path, data_set_name: str) -> Path:
    """
    Read each parquet file as a separate table, save to a database file, and perform cross-table analysis using Ibis.

    Args:
        pq_path (Path): Path to the directory containing parquet files.
        data_set_name (str): Name of the dataset, used for the database filename.

    This function demonstrates how to work with multiple tables in a database using Ibis,
    including joining tables and performing cross-table analysis where appropriate.
    """
    ibis.options.interactive = True

    db_path = read_parquet_files_as_tables_ibis(pq_path, data_set_name)

    if not db_path:
        return

    print(f"\nReconnecting to the saved database: {db_path}")
    conn = ibis.duckdb.connect(str(db_path))

    table_names = conn.list_tables()
    tables = {name: conn.table(name) for name in table_names}

    print(f"Found {len(tables)} tables in the database.")

    print("\n" + "=" * 80)
    print("CROSS-TABLE ANALYSIS")
    print("=" * 80)

    print("\nTable row counts:")
    for name, table in sorted(tables.items()):
        try:
            count = table.count().execute()
            print(f"  {name}: {count} rows")
        except Exception as e:
            print(f"  {name}: Error counting rows - {e}")

    print("\nTables with matching row counts (potential join candidates):")
    row_counts = {}
    for name, table in tables.items():
        try:
            count = table.count().execute()
            if count not in row_counts:
                row_counts[count] = []
            row_counts[count].append(name)
        except Exception:
            pass

    for count, table_names in sorted(row_counts.items()):
        if len(table_names) > 1:
            print(f"  Row count {count}: {', '.join(table_names)}")

    conn.con.close()

    return db_path

def upload_to_huggingface_hub(
    data_set_path: Path, 
    repo_id: str = "pyrovelocity/fixtures",
) -> RepoUrl:
    """
    Uploads a directory to the Hugging Face Hub.
    Assumes authentication.

    - https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication

    Args:
        data_set_path (Path): Path to the directory to upload.
        data_set_name (str): Name of the dataset, used for the repository name.
    """
    api = HfApi()
    repourl = api.create_repo(
        repo_id=repo_id, 
        private=False,
        repo_type="dataset", 
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=data_set_path,
        repo_type="dataset",
        revision="main",
        private=False,
        num_workers=max(os.cpu_count() - 2, 2),
        print_report=True,
        print_report_every=10,
    )
    return repourl


if __name__ == "__main__":
    file_path = Path("data")
    data_set_name = "postprocessed_pancreas_50_7"

    data_set_path = file_path / data_set_name
    h5ad_file = data_set_path / f"{data_set_name}.h5ad"
    pq_path = data_set_path / f"{data_set_name}.pqdata"

    adata, adata_pq = convert_h5ad_to_pqdata(h5ad_file, pq_path)

    repourl = upload_to_huggingface_hub(data_set_path)

    print(repourl)

    print("\n" + "=" * 80)
    print("USING DUCKDB DIRECTLY")
    print("=" * 80)
    read_parquet_files_duckdb(pq_path, data_set_name)

    print("\n" + "=" * 80)
    print("USING IBIS WITH DUCKDB BACKEND")
    print("=" * 80)
    analyze_parquet_database_ibis(pq_path, data_set_name)

    print("\n" + "=" * 80)
    print("USING DUCKDB VIEW REFERENCES")
    print("=" * 80)
    read_parquet_files_duckdb_views(pq_path, data_set_name)

    print("\n" + "=" * 80)
    print("USING IBIS WITH DUCKDB VIEW REFERENCES")
    print("=" * 80)
    read_parquet_files_as_views_ibis(pq_path, data_set_name)

    print("\n" + "=" * 80)
    print("DATABASE SIZE COMPARISON")
    print("=" * 80)
    duckdb_db_path = Path(f"{data_set_name}_duckdb.db")
    ibis_db_path = Path(f"{data_set_name}_ibis.db")
    duckdb_views_db_path = Path(f"{data_set_name}_duckdb_views.db")
    ibis_views_db_path = Path(f"{data_set_name}_ibis_views.db")

    db_paths = [
        ("DuckDB (tables)", duckdb_db_path),
        ("Ibis (tables)", ibis_db_path),
        ("DuckDB (views)", duckdb_views_db_path),
        ("Ibis (views)", ibis_views_db_path),
    ]

    print("Database sizes:")
    for name, path in db_paths:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            size_kb = path.stat().st_size / 1024
            if size_mb < 1:
                print(f"  {name}: {size_kb:.2f} KB")
            else:
                print(f"  {name}: {size_mb:.2f} MB")

    print("\nVerifying databases contain queryable tables/views...")

    for name, path in db_paths:
        if path.exists():
            try:
                if "Ibis" in name:
                    conn = ibis.duckdb.connect(str(path))
                    tables = conn.list_tables()
                    print(f"  {name} contains {len(tables)} queryable tables/views")
                    conn.con.close()
                else:
                    conn = duckdb.connect(str(path))
                    tables = conn.execute("SHOW TABLES").fetchall()
                    print(f"  {name} contains {len(tables)} queryable tables/views")
                    conn.close()
            except Exception as e:
                print(f"  Error querying {name}: {e}")
