import os

import anyio
import appdirs
import httpx
from anyio import Semaphore
from beartype import beartype
from beartype.typing import List
from beartype.typing import Tuple
from diskcache import Cache
from returns.result import Failure
from returns.result import Result
from returns.result import Success

from pyrovelocity.logging import configure_logging


__all__ = ["fetch_gene_sequences"]

logger = configure_logging(__name__)

CONCURRENCY_LIMIT = 100


@beartype
async def fetch_sequence(
    gene_id: str,
    cache: Cache,
    client: httpx.AsyncClient,
    semaphore: Semaphore,
) -> Result[str, str]:
    """
    Fetch the genomic sequence for a given gene or transcript ID from Ensembl.

    Results are cached to avoid redundant requests.

    Args:
        gene_id (str): Ensembl gene or transcript ID.
        cache (Cache): DiskCache object for caching results.

    Returns:
        Result[str, str]: A result object encapsulating success or failure.
    """
    async with semaphore:
        if gene_id in cache:
            logger.info(f"Retrieved from cache: {gene_id}")
            return Success(cache[gene_id])

        url = f"https://rest.ensembl.org/sequence/id/{gene_id}?type=genomic"
        headers = {"Content-Type": "text/x-fasta"}
        response = await client.get(url, headers=headers)

        if response.status_code == 200:
            rate_limit_remaining = int(
                response.headers.get("x-ratelimit-remaining", 0)
            )
            if rate_limit_remaining % 1000 == 0:
                logger.warning(f"Rate limit remaining: {rate_limit_remaining}")

            sequence = response.text
            cache[gene_id] = sequence
            logger.info(f"Fetched and cached: {gene_id}")
            return Success(sequence)
        else:
            error_message = f"Failed to fetch {gene_id}: {response.status_code}"
            logger.error(error_message)
            return Failure(error_message)


async def main(gene_id_list: List[str], cache: Cache) -> List[Result[str, str]]:
    results = []
    semaphore = Semaphore(CONCURRENCY_LIMIT)
    async with httpx.AsyncClient(
        timeout=10.0
    ) as client, anyio.create_task_group() as tg:
        for gene_id in gene_id_list:
            tg.start_soon(
                fetch_and_log_sequence,
                gene_id,
                cache,
                client,
                results,
                semaphore,
            )
    return results


@beartype
async def fetch_and_log_sequence(
    gene_id: str,
    cache: Cache,
    client: httpx.AsyncClient,
    results: List,
    semaphore: Semaphore,
):
    logger.info(f"Starting fetch for {gene_id}")
    result = await fetch_sequence(gene_id, cache, client, semaphore)
    results.append(result)
    logger.info(f"Completed fetch for {gene_id}")


@beartype
def process_results(
    results: List[Result[str, str]]
) -> Tuple[List[str], List[str]]:
    """
    Separates the successful results from the failures.

    Args:
        results (List[Result[str, str]]): List of Result objects.

    Returns:
        Tuple[List[str], List[str]]:
            A tuple containing a list of successful sequences and
            a list of error messages.
    """
    successes = [
        result.unwrap() for result in results if isinstance(result, Success)
    ]
    failures = [
        result.failure() for result in results if isinstance(result, Failure)
    ]
    return (successes, failures)


@beartype
def fetch_gene_sequences(
    gene_id_list: List[str],
    cache_path: str = "./.ensembl",
    fasta_file_name: str = "genomic_sequences",
):
    """

    Example:
    >>> # xdoctest: +SKIP
    >>> gene_id_list = [
    ...   'ENSG00000121410',
    ...   'ENSG00000175899',
    ...   'ENSG00000166535',
    ...   # 'ENSG00000128274',
    >>> ]
    >>> print(gene_id_list)
    >>> sequences, _ = fetch_gene_sequences(gene_id_list)
    >>> print(sequences)
    """
    cache_dir = appdirs.user_cache_dir("pyrovelocity")
    os.makedirs(cache_dir, exist_ok=True)
    fasta_file_path = os.path.join(cache_dir, f"{fasta_file_name}.fasta")

    cache = Cache(cache_path)
    try:
        sequences = anyio.run(main, gene_id_list, cache, backend="trio")
        successes, failures = process_results(sequences)

        with open(fasta_file_path, "w") as fasta_file:
            for sequence in successes:
                fasta_file.write(f"{sequence}")

        logger.info(f"Sequences saved to {fasta_file_path}")
    finally:
        cache.close()

    return successes, failures
