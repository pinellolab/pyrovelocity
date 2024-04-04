import os
from typing import List

import appdirs
from beartype import beartype
from plumbum import local

from pyrovelocity.logging import configure_logging


logger = configure_logging(__name__)


def is_aria2c_installed() -> bool:
    """Check if aria2c is installed."""
    aria2c = local.get("aria2c", None)
    return aria2c is not None


@beartype
def download_with_aria2c(urls: List[str], cache_dir: str):
    """
    Use aria2c to download files in parallel.
    """
    aria2c = local["aria2c"]
    aria2c_cmd = aria2c[
        "-x",
        "16",
        "-s",
        "16",
        "-j",
        "16",
        "-k",
        "1M",
        "-d",
        cache_dir,
        "--continue",
    ]

    for url in urls:
        try:
            logger.info(f"Downloading {url} to {cache_dir}")
            aria2c_cmd(url)
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")


@beartype
def download(urls: List[str], subdir: str = "pyrovelocity"):
    """
    Download all URLs in the provided list `urls` in parallel, supporting resume.
    Uses aria2c if available, otherwise logs an error.

    Args:
        urls: List of URLs to download.

    Example:
        >>> # xdoctest: +SKIP
        >>> from pyrovelocity.io.fetch import download
        >>> urls = [
        ...     "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
        ...     "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz",
        ... ]
        >>> download(urls)

        >>> # xdoctest: +SKIP
        >>> from pyrovelocity.io.fetch import download
        >>> urls = [
        ...     "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
        ...     "https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
        ... ]
        >>> download(urls)
    """
    if not is_aria2c_installed():
        logger.error(
            "aria2c is not installed. Please install aria2c to proceed with downloads."
        )
        return

    try:
        cache_dir = os.path.join(appdirs.user_cache_dir(), subdir)
        os.makedirs(cache_dir, exist_ok=True)
        download_with_aria2c(urls, cache_dir)
    except KeyboardInterrupt:
        logger.info(
            f"Discontinuing download of\n\t{urls} to\n\t{cache_dir}\n\n"
        )
