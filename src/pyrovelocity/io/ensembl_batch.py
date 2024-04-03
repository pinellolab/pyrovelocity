import json
import os
import random
from os import PathLike
from pathlib import Path

import anyio
import appdirs
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from anyio import Semaphore
from beartype import beartype
from beartype.typing import Dict
from beartype.typing import List
from beartype.typing import Optional
from diskcache import Cache
from httpx import AsyncClient
from returns.result import Failure
from returns.result import Result
from returns.result import Success

from pyrovelocity.logging import configure_logging


__all__ = ["fetch_gene_sequences_batch"]

logger = configure_logging(__name__)

CONCURRENCY_LIMIT = 15

BatchResult = List[Result[dict, str]]


@beartype
async def backoff(retry_count: int):
    delay = min(60, (2**retry_count) + random.uniform(0, 1))
    print(f"Waiting for {delay} seconds before retrying...")
    await anyio.sleep(delay)


@beartype
async def fetch_sequences_batch(
    gene_ids: List[str],
    client: AsyncClient,
    retry_count: int = 0,
) -> BatchResult:
    """
    Fetch a batch of genomic sequences for a given list of gene or transcript IDs from Ensembl.

    Args:
        gene_ids (List[str]): List of Ensembl gene or transcript IDs.
        client (httpx.AsyncClient): HTTP client for making requests.

    Returns:
        BatchResult: A list of result objects encapsulating success or failure.
    """
    url = "https://rest.ensembl.org/sequence/id"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = json.dumps({"ids": gene_ids})
    max_retries = 3

    try:
        response = await client.post(url, data=payload, headers=headers)
        if response.status_code == 200:
            return [Success(item) for item in response.json()]
        elif response.status_code == 429 and retry_count < max_retries:
            await backoff(retry_count)
            return await fetch_sequences_batch(
                gene_ids, client, retry_count + 1
            )
        elif response.status_code == 404:
            logger.error(f"404 Not Found for IDs: {', '.join(gene_ids)}")
            return [
                Failure(f"404 Not Found for ID: {gene_id}")
                for gene_id in gene_ids
            ]
        else:
            error_message = f"Failed to fetch batch: {response.status_code}"
            logger.error(error_message)
            return [Failure(error_message)] * len(gene_ids)
    except httpx.ReadTimeout:
        if retry_count < max_retries:
            logger.warning(
                f"Timeout encountered. Retrying... Attempt {retry_count+1}/{max_retries}"
            )
            await backoff(retry_count)
            return await fetch_sequences_batch(
                gene_ids, client, retry_count + 1
            )
        else:
            logger.error("Max retries reached. Failing...")
            return [Failure("ReadTimeout")] * len(gene_ids)


@beartype
async def main(
    gene_id_list: List[str],
    cache: Cache,
    query_batch_size: int,
) -> pa.Table:
    """
    Checks the cache for existing gene sequences and fetches the missing
    ones in batches.
    """
    cache_hits = [gene_id for gene_id in gene_id_list if gene_id in cache]
    to_fetch = list(set(gene_id_list) - set(cache_hits))
    fetched_results: List[Dict] = []

    semaphore = Semaphore(CONCURRENCY_LIMIT)

    async with httpx.AsyncClient(
        timeout=30.0
    ) as client, anyio.create_task_group() as tg:
        for i in range(0, len(to_fetch), query_batch_size):
            batch_ids = to_fetch[i : i + query_batch_size]
            tg.start_soon(
                fetch_sequences_batch_and_log,
                batch_ids,
                client,
                fetched_results,
                cache,
                semaphore,
            )

    logger.info(
        f"Cache hits: {len(cache_hits)}, New fetches: {len(fetched_results)}"
    )
    all_results = [cache[gene_id] for gene_id in cache_hits] + fetched_results

    table = pa.Table.from_pylist(all_results)
    return table


@beartype
async def fetch_sequences_batch_and_log(
    batch_ids: List[str],
    client: AsyncClient,
    results: List[Dict],
    cache: Cache,
    semaphore: Semaphore,
):
    """
    Fetches a batch of gene sequences and logs the operation.
    Updates the cache and results list.
    """
    async with semaphore:
        batch_results = await fetch_sequences_batch(batch_ids, client)
        for result in batch_results:
            if isinstance(result, Success):
                item = result.unwrap()
                cache[item["id"]] = item
                results.append(item)
            elif isinstance(result, Failure):
                logger.error(f"Fetch failed: {result.failure()}")
        logger.info(f"Processed batch: {len(batch_results)} sequences")


@beartype
def fetch_gene_sequences_batch(
    gene_id_list: List[str],
    cache_path: str = "./.ensembl",
    parquet_file_name: str = "genomic_sequences",
    query_batch_size: Optional[int] = 2,
) -> PathLike:
    """
    Fetch gene sequences in batches and save to a Parquet file.

    Args:
        gene_id_list (List[str]): List of gene IDs to fetch.
        cache_path (str): Path to the cache directory.
        parquet_file_name (str): Name of the output Parquet file.
        query_batch_size (int, optional): Number of IDs to query in each batch.

    Example:
    >>> # xdoctest: +SKIP
    >>> import ibis
    >>> import appdirs
    >>> from pyrovelocity.io.ensembl_batch import fetch_gene_sequences_batch
    ...
    >>> gene_ids_example = ["ENST00000398004", "ENST00000357258", "ENST00000396444"]
    ...
    >>> fetch_gene_sequences_batch(gene_ids_example)
    ...
    >>> parquet_file_path = appdirs.user_cache_dir("pyrovelocity") + "/genomic_sequences.parquet"
    >>> print(parquet_file_path)
    ...
    >>> t = ibis.read_parquet(parquet_file_path)
    >>> df = t.limit(3).to_pandas()
    >>> print(type(t))
    >>> print(t)
    >>> rs = t.select(["desc"]).filter(t["id"].isin(["ENST00000398004"])).execute()
    >>> print(rs['desc'].iloc[0])
    chromosome:GRCh38:12:68746176:68781468:1
    """
    cache_dir = appdirs.user_cache_dir("pyrovelocity")
    os.makedirs(cache_dir, exist_ok=True)
    parquet_file_path = os.path.join(cache_dir, f"{parquet_file_name}.parquet")

    cache = Cache(directory=cache_path, size_limit=int(4e9))
    try:
        table = anyio.run(
            main, gene_id_list, cache, query_batch_size, backend="trio"
        )
        pq.write_table(table, parquet_file_path)

        logger.info(f"Data saved to {parquet_file_path}")
    finally:
        cache.close()

    return Path(parquet_file_path)
