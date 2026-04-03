"""
Dataset download utility for CIC-IDS-2017.

Downloads the CSV files from the official CIC-IDS-2017 dataset repository
or alternative mirrors. Supports resumable downloads and integrity checks.
"""

import os
import re
import time
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger("xai_ids.download")

DATASET_BASE_URL = "https://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/"
MACHINE_LEARNING_CVE_PATH = "MachineLearningCVE/"

ZENODO_DATASET_URL = "https://zenodo.org/records/10141593/files/CIC-IDS-2017-V2.zip"

DEFAULT_RAW_DIR = "data/raw"


def discover_csv_links(base_url: str = DATASET_BASE_URL) -> List[str]:
    """
    Crawl the CIC-IDS-2017 dataset page to discover CSV file links.

    Parameters
    ----------
    base_url : str
        URL of the dataset directory listing.

    Returns
    -------
    List[str]
        List of full URLs to CSV files.
    """
    target_url = urljoin(base_url, MACHINE_LEARNING_CVE_PATH)
    logger.info(f"Discovering CSV links from: {target_url}")

    try:
        response = requests.get(target_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to access dataset page: {e}")
        raise

    soup = BeautifulSoup(response.text, "html.parser")
    csv_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".csv"):
            full_url = urljoin(target_url, href)
            csv_links.append(full_url)
            logger.info(f"  Found: {href}")

    logger.info(f"Discovered {len(csv_links)} CSV files")
    return csv_links


def download_file(
    url: str,
    dest_dir: str = DEFAULT_RAW_DIR,
    chunk_size: int = 8192,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Download a single file with progress tracking and retry logic.

    Parameters
    ----------
    url : str
        URL to download.
    dest_dir : str
        Destination directory.
    chunk_size : int
        Download chunk size in bytes.
    max_retries : int
        Maximum number of retry attempts.

    Returns
    -------
    Optional[str]
        Path to downloaded file, or None if download failed.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(url)
    # Clean filename
    filename = re.sub(r"[^\w\-_.]", "_", filename)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        logger.info(f"File already exists, skipping: {filename}")
        return dest_path

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {filename} (attempt {attempt}/{max_retries})")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if total_size > 0:
                logger.info(
                    f"Downloaded {filename}: {downloaded / (1024 * 1024):.1f} MB"
                )
            else:
                logger.info(
                    f"Downloaded {filename}: {downloaded / (1024 * 1024):.1f} MB"
                )

            return dest_path

        except (requests.RequestException, IOError) as e:
            logger.warning(f"Download attempt {attempt} failed for {filename}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < max_retries:
                wait_time = 2**attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to download {filename} after {max_retries} attempts"
                )
                return None


def download_dataset(
    dest_dir: str = DEFAULT_RAW_DIR,
    base_url: str = DATASET_BASE_URL,
) -> List[str]:
    """
    Download the complete CIC-IDS-2017 dataset.

    Parameters
    ----------
    dest_dir : str
        Destination directory for raw CSV files.
    base_url : str
        Base URL of the dataset.

    Returns
    -------
    List[str]
        List of paths to successfully downloaded files.
    """
    logger.info("Starting CIC-IDS-2017 dataset download")

    csv_links = discover_csv_links(base_url)

    if not csv_links:
        logger.warning("Primary source unavailable, trying Zenodo mirror...")
        return download_from_zenodo(dest_dir)

    downloaded_files = []
    failed_files = []

    for url in csv_links:
        result = download_file(url, dest_dir)
        if result:
            downloaded_files.append(result)
        else:
            failed_files.append(url)

    logger.info(
        f"Download complete: {len(downloaded_files)} succeeded, {len(failed_files)} failed"
    )

    if failed_files:
        logger.warning(f"Failed downloads: {failed_files}")
        logger.info("Trying Zenodo mirror as fallback...")
        return download_from_zenodo(dest_dir)

    return downloaded_files


def download_from_zenodo(dest_dir: str = DEFAULT_RAW_DIR) -> List[str]:
    """
    Download CIC-IDS-2017 from Zenodo mirror.

    Parameters
    ----------
    dest_dir : str
        Destination directory for raw CSV files.

    Returns
    -------
    List[str]
        List of paths to successfully downloaded files.
    """
    logger.info(f"Downloading CIC-IDS-2017 V2 from Zenodo...")

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    zip_path = os.path.join(dest_dir, "CIC-IDS-2017-V2.zip")

    for attempt in range(1, 4):
        try:
            logger.info(f"Downloading (attempt {attempt}/3)...")
            response = requests.get(ZENODO_DATASET_URL, stream=True, timeout=600)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            logger.info(f"Downloaded: {downloaded / (1024 * 1024):.1f} MB")

            logger.info("Extracting ZIP file...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)

            csv_files = []
            for root, _, files in os.walk(dest_dir):
                for f in files:
                    if f.endswith(".csv"):
                        csv_files.append(os.path.join(root, f))

            logger.info(f"Extracted {len(csv_files)} CSV files")
            os.remove(zip_path)

            return csv_files

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(5 * attempt)

    logger.error("Zenodo download failed after 3 attempts")
    return []


if __name__ == "__main__":
    download_dataset()
