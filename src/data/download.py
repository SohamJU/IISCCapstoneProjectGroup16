"""Dataset downloading utilities."""

from pathlib import Path

import kagglehub  # type: ignore[import-untyped]
from src.config.data import AMAZON_KAGGLE_DATASET, AMAZON_RAW_DATA_DIR, AMAZON_RAW_DATA_FILENAME


def download_electronics_reviews(force_download: bool = False) -> Path:
    """Download Amazon Electronics ratings dataset from Kaggle.

    Downloads the dataset to the configured raw data directory and renames the
    downloaded CSV file to the configured raw data filename.

    Args:
        force_download (bool): If True, download the dataset even if it already exists.

    Returns:
        pathlib.Path: Path to the downloaded dataset file.
    """
    AMAZON_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    target_filepath = AMAZON_RAW_DATA_DIR / AMAZON_RAW_DATA_FILENAME
    if target_filepath.exists() and not force_download:
        print(f"Dataset already exists at {target_filepath}")
        return target_filepath

    download_path = kagglehub.dataset_download(
        AMAZON_KAGGLE_DATASET,
        output_dir=AMAZON_RAW_DATA_DIR,
        force_download=force_download,
    )
    print(f"Dataset downloaded to {download_path}")
    
    # Rename downloaded csv (with any name) to the target filename
    downloaded_file = next((f for f in AMAZON_RAW_DATA_DIR.iterdir() if f.is_file() and f.suffix == ".csv"), None)
    if downloaded_file:
        downloaded_file.rename(target_filepath)
        print(f"Dataset renamed to {target_filepath}")
        return target_filepath

    raise FileNotFoundError(f"Downloaded dataset file not found in {AMAZON_RAW_DATA_DIR}")


if __name__ == "__main__":
    download_electronics_reviews(force_download=False)