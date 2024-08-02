from pathlib import Path

import pytest

from pyrovelocity.analysis.transcriptome_properties import calculate_histograms
from pyrovelocity.analysis.transcriptome_properties import (
    extract_canonical_transcripts,
)
from pyrovelocity.analysis.transcriptome_properties import (
    generate_cumulative_histograms,
)
from pyrovelocity.analysis.transcriptome_properties import (
    generate_gene_length_polyA_db_for_species,
)
from pyrovelocity.analysis.transcriptome_properties import (
    list_polyN_subsequence_lengths_in_range,
)
from pyrovelocity.analysis.transcriptome_properties import (
    load_transcript_sequences,
)
from pyrovelocity.analysis.transcriptome_properties import (
    process_and_save_histograms_to_db,
)
from pyrovelocity.analysis.transcriptome_properties import save_gene_data_to_db
from pyrovelocity.analysis.transcriptome_properties import save_parameters_to_db


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.mark.parametrize(
    "func",
    [
        extract_canonical_transcripts,
        load_transcript_sequences,
        list_polyN_subsequence_lengths_in_range,
        calculate_histograms,
        generate_cumulative_histograms,
        save_gene_data_to_db,
        save_parameters_to_db,
        process_and_save_histograms_to_db,
    ],
)
def test_functions_callable(func):
    assert callable(func), f"{func.__name__} should be callable"


@pytest.mark.pyensembl
def test_generate_gene_length_polyA_db_for_species_runs_without_error(db_path):
    try:
        generate_gene_length_polyA_db_for_species(
            species="homo_sapiens",
            ensembl_release_version=110,
            db_path=db_path,
            num_genes=10,
        )
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.pyensembl
def test_db_created_after_generation(db_path):
    generate_gene_length_polyA_db_for_species(
        species="homo_sapiens",
        ensembl_release_version=110,
        db_path=db_path,
        num_genes=10,
    )
    assert (
        db_path.exists()
    ), "Database file should exist after running generation function"
