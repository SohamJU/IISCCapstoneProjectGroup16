"""Dataset downloading utilities."""

import kagglehub
import shutil
import pathlib
from src.config.data import AMAZON_KAGGLE_DATASET, AMAZON_RAW_DATA_DIR, AMAZON_RAW_DATA_FILENAME


def download_electronics_reviews(force_download: bool = False):
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
        force_download=force_download,
    )
    print(f"Dataset downloaded to {download_path}")
    
    # Copy downloaded csv (with any name) to the target filename
    download_dir = pathlib.Path(download_path)
    downloaded_file = next((f for f in download_dir.iterdir() if f.is_file() and f.suffix == ".csv"), None)
    if downloaded_file:
        shutil.copy2(downloaded_file, target_filepath)
        print(f"Dataset copied to {target_filepath}")


if __name__ == "__main__":
    download_electronics_reviews(force_download=False)