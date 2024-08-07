try:
    import ibis
    import polars as pl
    import pyarrow as pa
    from pyensembl import EnsemblRelease
except ImportError:
    raise ImportError(
        "Additional dependencies are required for generating "
        "the transcriptome properties database."
        "Please install them via the bioinformatics extra: "
        "`pip install pyrovelocity[bioinformatics]`."
    )

import numpy as np
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import List
from beartype.typing import Optional

from pyrovelocity.logging import configure_logging


__all__ = [
    "generate_gene_length_polyA_db_for_species",
]

logger = configure_logging(__name__)


@beartype
def extract_canonical_transcripts(
    ensembl_data: EnsemblRelease,
    num_genes: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Simulate SQL query to filter for high-quality transcripts and select one per gene.

    Args:
        ensembl_data: An instance of EnsemblRelease containing gene and transcript data.
        num_genes: Number of genes to process, for testing purposes.

    Returns:
        A dictionary mapping each gene name to a selected transcript sequence.
    """
    canonical_transcripts = {}

    sql_query = """
    SELECT gene_id, transcript_id, gene_name, strand
    FROM transcript
    WHERE transcript_biotype = 'protein_coding'
    AND transcript_support_level = 1
    AND source = 'ensembl_havana'
    ORDER BY gene_name, transcript_name ASC;
    """

    ensembl_query_results = ensembl_data.db.run_sql_query(sql_query)
    processed_genes = 0

    for row in ensembl_query_results:
        if num_genes and processed_genes >= num_genes:
            break

        gene_id, transcript_id, gene_name, strand = row
        if not gene_name:
            continue

        if gene_name not in canonical_transcripts:
            canonical_transcripts[gene_name] = {
                "gene_id": gene_id,
                "transcript_id": transcript_id,
                "strand": strand,
            }
            processed_genes += 1

    return canonical_transcripts


@beartype
def load_transcript_sequences(
    species: str = "homo_sapiens",
    ensembl_release_version: int = 110,
    polyA_threshold_length: int = 15,
    max_length: int = 50,
    num_genes: Optional[int] = None,
    use_canonical: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load transcript sequences for each gene for the specified species and
    Ensembl release version.

    For reference regarding the Ensembl release version number see:

    - https://ensembl.org/info/website/archives/
    - https://sep2019.archive.ensembl.org/info/data/ftp/ (release 98)
    - https://ensembl.org/info/data/ftp/ (latest release)

    Args:
        species: The species name (default is "homo_sapiens").
        ensembl_release_version: The Ensembl release version number.
        polyA_threshold_length:
            The minimum length of polyA motifs to include in the summary count.
        num_genes: Number of genes. Defaults to None, which means All.
            Primarily used for testing.
        use_canonical: Flag to indicate exclusive usage of canonical transcripts.

    Returns:
        A list of dictionaries with gene information and transcript sequences.
    """
    ensembl_data = EnsemblRelease(
        release=ensembl_release_version, species=species
    )
    ensembl_data.download()
    ensembl_data.index()

    canonical_transcripts = {}
    if use_canonical:
        canonical_transcripts = extract_canonical_transcripts(
            ensembl_data, num_genes
        )
        logger.info(
            f"Using canonical transcripts identified for "
            f"{len(canonical_transcripts)} genes\n\n"
        )

    gene_data = []
    processed_genes = 0

    genes_to_process = (
        canonical_transcripts.keys()
        if use_canonical
        else (gene.name for gene in ensembl_data.genes())
    )

    for gene_name in genes_to_process:
        if num_genes is not None and processed_genes >= num_genes:
            break

        try:
            if use_canonical:
                transcript_id = canonical_transcripts[gene_name][
                    "transcript_id"
                ]
                transcript = ensembl_data.transcript_by_id(transcript_id)
                gene = ensembl_data.gene_by_id(
                    canonical_transcripts[gene_name]["gene_id"]
                )
                seq = transcript.sequence if transcript else None
                exon_length = (
                    sum(len(exon) for exon in transcript.exons)
                    if transcript and transcript.exons
                    else 0
                )
                transcript_seqs = [seq] if seq else []
            else:
                gene = ensembl_data.gene_by_name(gene_name)
                transcript_seqs = [
                    transcript.sequence
                    for transcript in gene.transcripts
                    if transcript.sequence
                ]
                exon_length = 0

            gene_length = gene.end - gene.start + 1
            polyA_counts = [
                length
                for seq in transcript_seqs
                for length in list_polyN_subsequence_lengths_in_range(
                    seq, polyA_threshold_length, max_length
                )
            ]
            polyA_count_above_threshold = sum(
                count >= polyA_threshold_length for count in polyA_counts
            )

            intron_length = gene_length - exon_length
            intron_fraction = intron_length / gene_length

            gene_data.append(
                {
                    "gene_name": gene_name,
                    "transcript_id": transcript.id if use_canonical else "All",
                    "gene_length": gene_length,
                    "coding_length": exon_length,
                    "intron_length": intron_length,
                    "intron_fraction": intron_fraction,
                    f"polyA_count_above_{polyA_threshold_length}": polyA_count_above_threshold,
                    "num_transcripts": len(transcript_seqs),
                    "sequences": transcript_seqs,
                    "species": species,
                }
            )

            processed_genes += 1

        except ValueError as e:
            logger.warn(f"Skipping gene {gene_name} due to error: {e}")
            continue

    return gene_data


@beartype
def list_polyN_subsequence_lengths_in_range(
    sequence: str,
    min_length: int = 5,
    max_length: int = 50,
    nucleotide: str = "A",
) -> List[int]:
    """
    Analyze the given sequence for polyA repeat patterns and return their lengths.

    Args:
        sequence: The DNA sequence to analyze.
        min_length: The minimum repeat length to consider.
        max_length: The maximum repeat length to consider.
        nucleotide: The nucleotide to consider for repeat patterns.

    Returns:
        Lengths of found repeats.

    Examples:
        >>> analyze_sequence_patterns("AAAATTTTCCCCGGG", 2, 4)
        [4]
        >>> analyze_sequence_patterns("AAAATTTTACCCCAAAGGGAA", 2, 4)
        [4, 3, 2]
    """
    lengths = []
    current_length = 0

    for char in sequence:
        if char == nucleotide:
            current_length += 1
        else:
            if min_length <= current_length <= max_length:
                lengths.append(current_length)
            current_length = 0

    if min_length <= current_length <= max_length:
        lengths.append(current_length)

    return lengths


@beartype
def calculate_histograms(
    gene_data: List[Dict[str, Any]],
    min_length: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    """
    Calculate histograms for each gene based on its transcript sequences.

    Args:
        gene_data: A list containing gene information including transcript sequences.
        min_length: The minimum length of polyA sequences to consider.
        max_length: The maximum length of polyA sequences to consider.

    Returns:
        A list of dictionaries, each representing histogram data for a gene.
    """
    hist_data = []
    hist_bins = np.append(np.arange(min_length, max_length + 1) - 0.5, np.inf)
    for gene in gene_data:
        all_repeats = [
            repeat_length
            for seq in gene["sequences"]
            for repeat_length in list_polyN_subsequence_lengths_in_range(
                seq, min_length, max_length
            )
        ]
        hist, _ = (
            np.histogram(all_repeats, bins=hist_bins)
            if all_repeats
            else (np.zeros(len(hist_bins) - 1), None)
        )
        gene_name = gene["gene_name"]
        hist_data.append(
            {
                "gene_name": gene_name,
                **{
                    f"bin_{min_length + i}": hist_val
                    for i, hist_val in enumerate(hist)
                },
            }
        )
    return hist_data


@beartype
def generate_cumulative_histograms(
    hist_data: List[Dict[str, Any]],
    min_length: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    """
    Generate cumulative histograms where each bin represents the total number of sequences
    of that length or greater, using a direct approach with reversed cumulative sums.

    Args:
        hist_data: Histogram data for each gene, with bins keyed by 'bin_{length}'.
        min_length: The minimum length of sequences considered.
        max_length: The maximum length of sequences considered.

    Returns:
        A list of dictionaries, each representing the cumulative histogram data for a gene.
    """
    cum_hist_data = []
    for gene_hist in hist_data:
        counts = [
            gene_hist.get(f"bin_{i}", 0)
            for i in range(min_length, max_length + 1)
        ]

        cum_counts = np.cumsum(counts[::-1])[::-1]

        cum_hist_entry = {
            "gene_name": gene_hist["gene_name"],
            **{
                f"cum_bin_{i}": int(cum_val)
                for i, cum_val in zip(
                    range(min_length, max_length + 1), cum_counts
                )
            },
        }

        cum_hist_data.append(cum_hist_entry)

    return cum_hist_data


@beartype
def polars_to_pyarrow(
    df: pl.DataFrame,
) -> pa.Table:
    """
    Convert a Polars DataFrame to a PyArrow Table.
    This wrapper function only exists for type-checking purposes.
    Otherwise, this could just be executed in-line.

    Args:
        df: The Polars DataFrame to convert.

    Returns:
        A PyArrow Table representation of the input Polars DataFrame.
    """
    return df.to_arrow()


@beartype
def save_gene_data_to_db(
    gene_data: List[Dict[str, Any]],
    species: str,
    db_path: str,
    save_sequences: bool = False,
) -> None:
    """
    Save gene information including gene length and count of long polyA
    sequences to database.

    Args:
        gene_data: A list of dictionaries containing gene information.
        species: The species name for database naming.
        db_path: Path to the database file.
        save_sequences:
            Flag to indicate whether to save sequences to the database.
            Default is False.
    """

    if save_sequences:
        gene_data_for_db = gene_data
    else:
        gene_data_for_db = [
            {k: v for k, v in gene.items() if k != "sequences"}
            for gene in gene_data
        ]
    df_gene_data = pl.DataFrame(gene_data_for_db)

    table_gene_data = df_gene_data.to_arrow()

    con = ibis.duckdb.connect(db_path)
    con.create_table(
        f"{species}_gene_data", obj=table_gene_data, overwrite=True
    )


@beartype
def save_parameters_to_db(
    species: str,
    db_path: str,
    min_length: int,
    max_length: int,
    polyA_threshold_length: int,
    ensembl_release_version: int,
) -> None:
    """
    Save histogram calculation parameters to database.

    Args:
        species: The species name, used for database naming.
        db_path: Path to the database file.
        min_length: The minimum length of polyA sequences considered.
        max_length: The maximum length of polyA sequences considered.
        polyA_threshold_length:
            The threshold length for counting long polyA sequences.
        ensembl_release_version: The Ensembl release version.
    """
    parameters = {
        "species": species,
        "min_length": min_length,
        "max_length": max_length,
        "ensembl_release_version": ensembl_release_version,
        "polyA_threshold_length": polyA_threshold_length,
    }
    df_parameters = pl.DataFrame([parameters])

    table_parameters = df_parameters.to_arrow()

    con = ibis.duckdb.connect(db_path)
    if "parameters" in con.list_tables():
        con.insert("parameters", obj=parameters)
        logger.info(
            f"Appended parameters for {species} to the existing 'parameters' table."
        )
    else:
        con.create_table("parameters", obj=table_parameters, overwrite=True)
        logger.info(
            f"Created 'parameters' table and added parameters for {species}."
        )


@beartype
def process_and_save_histograms_to_db(
    sequences: List[Dict[str, Any]],
    species: str,
    db_path: str,
    min_length: int,
    max_length: int,
) -> None:
    """
    Process sequences to generate histograms and cumulative histograms,
    and save them into database as tables.

    Args:
        sequences: Loaded sequences and their metadata.
        species: The species name for database naming.
        db_path: Path to the database file.
        min_length: Minimum length of polyA sequences to consider.
        max_length: Maximum length of polyA sequences to consider.
        hist_bins: Binning information for histogram calculation.
    """
    hist_data = calculate_histograms(
        sequences,
        min_length,
        max_length,
    )
    cum_hist_data = generate_cumulative_histograms(
        hist_data=hist_data,
        min_length=min_length,
        max_length=max_length,
    )

    df_hist = pl.DataFrame(hist_data)
    df_cum_hist = pl.DataFrame(cum_hist_data)
    table_hist = df_hist.to_arrow()
    table_cum_hist = df_cum_hist.to_arrow()

    con = ibis.duckdb.connect(db_path)
    con.create_table(
        f"{species}_polyA_histogram",
        obj=table_hist,
        overwrite=True,
    )
    con.create_table(
        f"{species}_polyA_histogram_cumulative",
        obj=table_cum_hist,
        overwrite=True,
    )


def generate_gene_length_polyA_db_for_species(
    species: str = "homo_sapiens",
    ensembl_release_version: int = 110,
    db_path: str = "gene_length_polyA_motifs.ddb",
    num_genes: Optional[int] = 100,
    min_length: int = 5,
    max_length: int = 50,
    polyA_threshold_length: int = 8,
) -> None:
    """
    Main function to orchestrate loading of sequences, calculation of histograms,
    and saving data to the database.

    The database should only need to be regenerated when updating to a new
    Ensembl release or modifying parameters. The procedure used to generate the
    database is nearly equivalent to that shown in the Examples below except
    that num_genes is set to None in order to include all genes that meet
    the transcript quality selection criteria associated to Ensembl canonical
    or MANE transcript flags.

    Args:
        species: Species name.
        ensembl_release_version:
            Version of Ensembl release to use for sequence loading.
        db_path: Path to the database file.
        num_genes: Number of genes. Defaults to 100. Uses all genes if None.
        min_length: Minimum length to include in polyN histogram.
        max_length: Maximum length to include in polyN histogram.
        polyA_threshold_length:
            Minimum length to include in summary of number of polyA motifs in
            each transcript.

    Examples:
        >>> # xdoctest: +SKIP
        >>> # define fixtures
        >>> try:
        >>>     tmp = getfixture("tmp_path")
        >>> except NameError:
        >>>     import tempfile
        >>>     tmp = tempfile.TemporaryDirectory().name
        >>> ensembl_release_version = 110
        >>> db_name = f"test_gene_length_polyA_motifs_ensembl_{ensembl_release_version}.ddb"
        >>> db_path = str(tmp) + "/" + db_name
        >>> print(db_path)
        >>> # generate transcriptome properties database for multiple species
        >>> species_list = ["homo_sapiens", "mus_musculus"]
        >>> for species in species_list:
        >>>     generate_gene_length_polyA_db_for_species(
        ...         species=species,
        ...         ensembl_release_version=ensembl_release_version,
        ...         db_path=db_path,
        ...         num_genes=50,
        >>>     )
    """
    logger.info(
        f"Processing {species} data for Ensembl release "
        f"{ensembl_release_version} and saving to\n"
        f"\t{db_path}\n\n"
    )

    sequences = load_transcript_sequences(
        species=species,
        ensembl_release_version=ensembl_release_version,
        polyA_threshold_length=polyA_threshold_length,
        num_genes=num_genes,
    )
    save_gene_data_to_db(
        gene_data=sequences,
        species=species,
        db_path=db_path,
    )
    process_and_save_histograms_to_db(
        sequences=sequences,
        species=species,
        db_path=db_path,
        min_length=min_length,
        max_length=max_length,
    )
    save_parameters_to_db(
        species=species,
        db_path=db_path,
        min_length=min_length,
        max_length=max_length,
        polyA_threshold_length=polyA_threshold_length,
        ensembl_release_version=ensembl_release_version,
    )
